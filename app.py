import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import calendar
import io
from datetime import datetime

# ★★★ バージョン情報 ★★★
APP_VERSION = "proto.1.0"

# --- ヘルパー関数: サマリー作成 ---
def _create_summary(schedule_df, staff_info_dict, year, month, event_units):
    num_days = calendar.monthrange(year, month)[1]
    days = list(range(1, num_days + 1))
    daily_summary = []
    for d in days:
        day_info = {}
        work_staff_ids = schedule_df[schedule_df[d] == '出']['職員番号']
        half_day_staff_ids = [s for s, dates in st.session_state.requests_half.items() if d in dates]
        total_workers = sum(0.5 if sid in half_day_staff_ids else 1 for sid in work_staff_ids)
        day_info['日'] = d
        day_info['曜日'] = ['月','火','水','木','金','土','日'][calendar.weekday(year, month, d)]
        day_info['出勤者総数'] = total_workers
        day_info['PT'] = sum(0.5 if sid in half_day_staff_ids else 1 for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '理学療法士')
        day_info['OT'] = sum(0.5 if sid in half_day_staff_ids else 1 for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '作業療法士')
        day_info['ST'] = sum(0.5 if sid in half_day_staff_ids else 1 for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '言語聴覚士')
        day_info['役職者'] = sum(0.5 if sid in half_day_staff_ids else 1 for sid in work_staff_ids if pd.notna(staff_info_dict[sid]['役職']))
        day_info['回復期'] = sum(0.5 if sid in half_day_staff_ids else 1 for sid in work_staff_ids if staff_info_dict[sid].get('役割1') == '回復期専従')
        day_info['地域包括'] = sum(0.5 if sid in half_day_staff_ids else 1 for sid in work_staff_ids if staff_info_dict[sid].get('役割1') == '地域包括専従')
        day_info['外来'] = sum(0.5 if sid in half_day_staff_ids else 1 for sid in work_staff_ids if staff_info_dict[sid].get('役割1') == '外来PT')
        if calendar.weekday(year, month, d) != 6:
            pt_units = sum(int(staff_info_dict[sid]['1日の単位数']) * (0.5 if sid in half_day_staff_ids else 1) for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '理学療法士')
            ot_units = sum(int(staff_info_dict[sid]['1日の単位数']) * (0.5 if sid in half_day_staff_ids else 1) for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '作業療法士')
            st_units = sum(int(staff_info_dict[sid]['1日の単位数']) * (0.5 if sid in half_day_staff_ids else 1) for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '言語聴覚士')
            day_info['PT単位数'] = pt_units
            day_info['OT単位数'] = ot_units
            day_info['ST単位数'] = st_units
            day_info['PT+OT単位数'] = pt_units + ot_units
            day_info['特別業務単位数'] = event_units.get(d, 0)
        else:
            day_info['PT単位数'] = '-'; day_info['OT単位数'] = '-'; day_info['ST単位数'] = '-';
            day_info['PT+OT単位数'] = '-'; day_info['特別業務単位数'] = '-'
        daily_summary.append(day_info)
    return pd.DataFrame(daily_summary)

# --- 勤務表DataFrame作成ヘルパー ---
def _create_schedule_df(shifts_values, staff, days, staff_df, requests_x, requests_tri, requests_paid, requests_special):
    schedule_data = {}
    for s in staff:
        row = []
        s_requests_x = requests_x.get(s, [])
        s_requests_tri = requests_tri.get(s, [])
        s_requests_paid = requests_paid.get(s, [])
        s_requests_special = requests_special.get(s, [])
        for d in days:
            if shifts_values.get((s, d), 0) == 0:
                if d in s_requests_x: row.append('×')
                elif d in s_requests_tri: row.append('△')
                elif d in s_requests_paid: row.append('有')
                elif d in s_requests_special: row.append('特')
                else: row.append('-')
            else: row.append('')
        schedule_data[s] = row
    schedule_df = pd.DataFrame.from_dict(schedule_data, orient='index', columns=days)
    schedule_df = schedule_df.reset_index().rename(columns={'index': '職員番号'})
    staff_map = staff_df.set_index('職員番号')
    schedule_df.insert(1, '職員名', schedule_df['職員番号'].map(staff_map['職員名']))
    schedule_df.insert(2, '職種', schedule_df['職員番号'].map(staff_map['職種']))
    return schedule_df

# --- メインのソルバー関数 ---
def solve_final_model(staff_df, requests_df, year, month, 
                      target_pt, target_ot, target_st, tolerance,
                      event_units, tri_penalty_weight):
    num_days = calendar.monthrange(year, month)[1]; days = list(range(1, num_days + 1)); staff = staff_df['職員番号'].tolist()
    staff_info = staff_df.set_index('職員番号').to_dict('index')
    sundays = [d for d in days if calendar.weekday(year, month, d) == 6]; weekdays = [d for d in days if d not in sundays]
    managers = [s for s in staff if pd.notna(staff_info[s]['役職'])]; pt_staff = [s for s in staff if staff_info[s]['職種'] == '理学療法士']
    ot_staff = [s for s in staff if staff_info[s]['職種'] == '作業療法士']; st_staff = [s for s in staff if staff_info[s]['職種'] == '言語聴覚士']
    kaifukuki_staff = [s for s in staff if staff_info[s].get('役割1') == '回復期専従']; kaifukuki_pt = [s for s in kaifukuki_staff if staff_info[s]['職種'] == '理学療法士']
    kaifukuki_ot = [s for s in kaifukuki_staff if staff_info[s]['職種'] == '作業療法士']; gairai_staff = [s for s in staff if staff_info[s].get('役割1') == '外来PT']
    chiiki_staff = [s for s in staff if staff_info[s].get('役割1') == '地域包括専従']; sunday_off_staff = gairai_staff + chiiki_staff
    requests_x = {}; requests_tri = {}; requests_must_work = {};
    requests_paid = {}; requests_special = {}; st.session_state.requests_half = {};
    for index, row in requests_df.iterrows():
        staff_id = row['職員番号']
        if staff_id not in staff: continue
        requests_x[staff_id] = [d for d in days if str(d) in row and row.get(str(d)) == '×']
        requests_tri[staff_id] = [d for d in days if str(d) in row and row.get(str(d)) == '△']
        requests_must_work[staff_id] = [d for d in days if str(d) in row and row.get(str(d)) == '○']
        requests_paid[staff_id] = [d for d in days if str(d) in row and row.get(str(d)) == '有']
        requests_special[staff_id] = [d for d in days if str(d) in row and row.get(str(d)) == '特']
        st.session_state.requests_half[staff_id] = [d for d in days if str(d) in row and row.get(str(d)) in ['AM有', 'PM有']]
    model = cp_model.CpModel(); shifts = {}
    for s in staff:
        for d in days: shifts[(s, d)] = model.NewBoolVar(f'shift_{s}_{d}')
    for s in staff:
        num_paid_leave = len(requests_paid.get(s, [])); num_special_leave = len(requests_special.get(s, []))
        model.Add(sum(1 - shifts[(s, d)] for d in days) == 9 + num_paid_leave + num_special_leave)
    for s, dates in requests_must_work.items():
        for d in dates: model.Add(shifts[(s, d)] == 1)
    for s, dates in st.session_state.requests_half.items():
        for d in dates: model.Add(shifts[(s, d)] == 1)
    for s, dates in requests_x.items():
        for d in dates: model.Add(shifts[(s, d)] == 0)
    for s, dates in requests_paid.items():
        for d in dates: model.Add(shifts[(s, d)] == 0)
    for s, dates in requests_special.items():
        for d in dates: model.Add(shifts[(s, d)] == 0)
    for d in days: model.Add(sum(shifts[(s, d)] for s in managers) >= 1)
    for s in sunday_off_staff:
        for d in sundays: model.Add(shifts[(s, d)] == 0)
    for s in staff: model.Add(sum(shifts[(s, d)] for d in sundays) <= 2)
    penalties = []
    for s, req_days in requests_tri.items():
        if s in staff:
            for d in req_days: penalties.append(tri_penalty_weight * shifts[(s, d)])
    weeks_in_month = []; current_week = []
    for d in days:
        current_week.append(d)
        if calendar.weekday(year, month, d) == 5 or d == num_days: weeks_in_month.append(current_week); current_week = []
    for s_idx, s in enumerate(staff):
        all_requests = requests_x.get(s, []) + requests_tri.get(s, []) + requests_paid.get(s, []) + requests_special.get(s, [])
        for w_idx, week in enumerate(weeks_in_month):
            if sum(1 for d in week if d in all_requests) >= 3: continue
            num_holidays_in_week = sum(1 - shifts[(s, d)] for d in week)
            if len(week) == 7:
                violation = model.NewBoolVar(f'f_w_v_s{s_idx}_w{w_idx}'); model.Add(num_holidays_in_week < 2).OnlyEnforceIf(violation); model.Add(num_holidays_in_week >= 2).OnlyEnforceIf(violation.Not()); penalties.append(200 * violation)
            else:
                violation = model.NewBoolVar(f'p_w_v_s{s_idx}_w{w_idx}'); model.Add(num_holidays_in_week < 1).OnlyEnforceIf(violation); model.Add(num_holidays_in_week >= 1).OnlyEnforceIf(violation.Not()); penalties.append(25 * violation)
    for d in sundays:
        pt_on_sunday = sum(shifts[(s, d)] for s in pt_staff); ot_on_sunday = sum(shifts[(s, d)] for s in ot_staff); st_on_sunday = sum(shifts[(s, d)] for s in st_staff)
        total_pt_ot = pt_on_sunday + ot_on_sunday; total_diff = model.NewIntVar(-50, 50, f't_d_{d}'); model.Add(total_diff == total_pt_ot - (target_pt + target_ot)); abs_total_diff = model.NewIntVar(0, 50, f'a_t_d_{d}'); model.AddAbsEquality(abs_total_diff, total_diff); penalties.append(50 * abs_total_diff)
        pt_diff = model.NewIntVar(-30, 30, f'p_d_{d}'); model.Add(pt_diff == pt_on_sunday - target_pt); pt_penalty = model.NewIntVar(0, 30, f'p_p_{d}'); model.Add(pt_penalty >= pt_diff - tolerance); model.Add(pt_penalty >= -pt_diff - tolerance); penalties.append(40 * pt_penalty)
        ot_diff = model.NewIntVar(-30, 30, f'o_d_{d}'); model.Add(ot_diff == ot_on_sunday - target_ot); ot_penalty = model.NewIntVar(0, 30, f'o_p_{d}'); model.Add(ot_penalty >= ot_diff - tolerance); model.Add(ot_penalty >= -ot_diff - tolerance); penalties.append(40 * ot_penalty)
        st_diff = model.NewIntVar(-10, 10, f's_d_{d}'); model.Add(st_diff == st_on_sunday - target_st); abs_st_diff = model.NewIntVar(0, 10, f'a_s_d_{d}'); model.AddAbsEquality(abs_st_diff, st_diff); penalties.append(60 * abs_st_diff)
    for d in days:
        num_gairai_off = sum(1 - shifts[(s, d)] for s in gairai_staff); penalty = model.NewIntVar(0, len(gairai_staff), f'g_p_{d}'); model.Add(penalty >= num_gairai_off - 1); penalties.append(10 * penalty)
        model.Add(sum(shifts[(s, d)] for s in kaifukuki_staff) >= 1); pt_present = model.NewBoolVar(f'k_p_p_{d}'); ot_present = model.NewBoolVar(f'k_o_p_{d}'); model.Add(sum(shifts[(s, d)] for s in kaifukuki_pt) >= 1).OnlyEnforceIf(pt_present); model.Add(sum(shifts[(s, d)] for s in kaifukuki_pt) == 0).OnlyEnforceIf(pt_present.Not()); model.Add(sum(shifts[(s, d)] for s in kaifukuki_ot) >= 1).OnlyEnforceIf(ot_present); model.Add(sum(shifts[(s, d)] for s in kaifukuki_ot) == 0).OnlyEnforceIf(ot_present.Not()); penalties.append(5 * (1 - pt_present)); penalties.append(5 * (1 - ot_present))
    job_types = {'PT': pt_staff, 'OT': ot_staff, 'ST': st_staff};
    for job, members in job_types.items():
        if not members: continue
        avg_work_days = (num_days - 9) * len(members); target_per_weekday = avg_work_days / len(weekdays) if weekdays else 0
        for d in weekdays:
            actual = sum(shifts[(s, d)] for s in members); diff = model.NewIntVar(-len(members), len(members), f'd_{job}_{d}'); model.Add(diff == actual - round(target_per_weekday)); abs_diff = model.NewIntVar(0, len(members), f'a_d_{job}_{d}'); model.AddAbsEquality(abs_diff, diff); penalties.append(1 * abs_diff)
    total_weekday_units = sum(int(staff_info[s]['1日の単位数']) for s in staff) * (len(weekdays)) * (len(staff)-9)/len(staff); total_event_units = sum(event_units.values()); avg_residual_units = (total_weekday_units - total_event_units) / len(weekdays) if weekdays else 0
    for d in weekdays:
        provided_units = sum(shifts[(s, d)] * int(staff_info[s]['1日の単位数']) * (0.5 if s in st.session_state.requests_half.get(s, []) and d in st.session_state.requests_half[s] else 1.0) for s in staff); event_unit = event_units.get(d, 0); residual_units = model.NewIntVar(-2000, 2000, f'r_{d}'); model.Add(residual_units == provided_units - event_unit); diff = model.NewIntVar(-2000, 2000, f'u_d_{d}'); model.Add(diff == residual_units - round(avg_residual_units)); abs_diff = model.NewIntVar(0, 2000, f'a_u_d_{d}'); model.AddAbsEquality(abs_diff, diff); penalties.append(2 * abs_diff)
    model.Minimize(sum(penalties))
    solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds = 60.0; status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        shifts_values = {(s, d): solver.Value(shifts[(s, d)]) for s in staff for d in days}
        schedule_df = _create_schedule_df(shifts_values, staff, days, staff_df, requests_x, requests_tri, requests_paid, requests_special)
        temp_work_df = schedule_df.replace({'×': '休', '-': '休', '△': '休', '有': '休', '特': '休', '': '出'})
        summary_df = _create_summary(temp_work_df, staff_info, year, month, event_units)
        message = f"求解ステータス: **{solver.StatusName(status)}** (ペナルティ合計: **{round(solver.ObjectiveValue())}**)"
        return True, schedule_df, summary_df, message
    else:
        message = f"致命的なエラー: ハード制約が矛盾しているため、勤務表を作成できませんでした。({solver.StatusName(status)})"
        return False, pd.DataFrame(), pd.DataFrame(), message

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title('リハビリテーション科 勤務表作成アプリ')

with st.expander("▼ 各種パラメータを設定する", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("対象年月とファイル")
        current_year = datetime.now().year
        year = st.number_input("年（西暦）", min_value=current_year - 5, max_value=current_year + 5, value=current_year)
        month = st.selectbox("月", options=list(range(1, 13)), index=datetime.now().month-1)
        st.markdown("---")
        staff_file = st.file_uploader("1. 職員一覧 (CSV)", type="csv")
        requests_file = st.file_uploader("2. 希望休一覧 (CSV)", type="csv")
    with c2:
        st.subheader("日曜日の出勤人数設定")
        c2_1, c2_2, c2_3 = st.columns(3)
        with c2_1: target_pt = st.number_input("PT目標", min_value=0, value=10, step=1)
        with c2_2: target_ot = st.number_input("OT目標", min_value=0, value=5, step=1)
        with c2_3: target_st = st.number_input("ST目標", min_value=0, value=3, step=1)
    with c3:
        st.subheader("緩和条件と優先度")
        tolerance = st.number_input("PT/OT許容誤差(±)", min_value=0, max_value=5, value=1, help="PT/OTの合計人数が目標通りなら、それぞれの人数がこの値までずれてもペナルティを課しません。")
        tri_penalty_weight = st.slider("準希望休(△)の優先度", min_value=0, max_value=20, value=8, help="値が大きいほど△希望が尊重されます。")

    st.markdown("---")
    st.subheader(f"{year}年{month}月のイベント設定（各日の特別業務単位数を入力）")
    event_units_input = {}
    num_days_in_month = calendar.monthrange(year, month)[1]
    first_day_weekday = calendar.weekday(year, month, 1)
    cal_cols = st.columns(7)
    weekdays_jp = ['月', '火', '水', '木', '金', '土', '日']
    for i, day_name in enumerate(weekdays_jp): cal_cols[i].markdown(f"<p style='text-align: center;'><b>{day_name}</b></p>", unsafe_allow_html=True)
    day_counter = 1
    for week in range(6):
        cols = st.columns(7)
        for day_of_week in range(7):
            if (week == 0 and day_of_week < first_day_weekday) or day_counter > num_days_in_month:
                continue
            with cols[day_of_week]:
                is_sunday = calendar.weekday(year, month, day_counter) == 6
                event_units_input[day_counter] = st.number_input(label=f"{day_counter}日", value=0, step=10, disabled=is_sunday, key=f"event_{year}_{month}_{day_counter}")
            day_counter += 1
        if day_counter > num_days_in_month: break
            
    st.markdown("---")
    create_button = st.button('勤務表を作成', type="primary", use_container_width=True)

if create_button:
    if staff_file is not None and requests_file is not None:
        try:
            staff_df = pd.read_csv(staff_file); requests_df = pd.read_csv(requests_file)
            if '職員名' not in staff_df.columns:
                staff_df['職員名'] = staff_df['職種'] + " " + staff_df['職員番号'].astype(str)
                st.info("職員一覧に「職員名」列がなかったため、仮の職員名を生成しました。")
            
            is_feasible, schedule_df, summary_df, message = solve_final_model(
                staff_df, requests_df, year, month,
                target_pt, target_ot, target_st, tolerance,
                event_units_input, tri_penalty_weight
            )
            st.info(message)
            if is_feasible:
                st.header("勤務表")
                num_days = calendar.monthrange(year, month)[1]
                summary_T = summary_df.drop(columns=['日', '曜日']).T
                summary_T.columns = list(range(1, num_days + 1))
                summary_processed = summary_T.reset_index().rename(columns={'index': '職員名'})
                summary_processed['職員番号'] = summary_processed['職員名'].apply(lambda x: f"_{x}")
                summary_processed['職種'] = "サマリー"
                summary_processed = summary_processed[['職員番号', '職員名', '職種'] + list(range(1, num_days + 1))]
                final_df_for_display = pd.concat([schedule_df, summary_processed], ignore_index=True)
                days_header = list(range(1, num_days + 1))
                weekdays_header = [ ['月','火','水','木','金','土','日'][calendar.weekday(year, month, d)] for d in days_header]
                final_df_for_display.columns = pd.MultiIndex.from_tuples([('職員情報', '職員番号'), ('職員情報', '職員名'), ('職員情報', '職種')] + list(zip(days_header, weekdays_header)))
                def style_table(df):
                    sunday_cols = [col for col in df.columns if col[1] == '日']
                    styler = df.style.set_properties(**{'text-align': 'center'})
                    for col in sunday_cols: styler = styler.set_properties(subset=[col], **{'background-color': '#fff0f0'})
                    return styler
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    schedule_df.to_excel(writer, sheet_name='勤務表', index=False)
                    summary_df.to_excel(writer, sheet_name='日別サマリー', index=False)
                excel_data = output.getvalue()
                st.download_button(label="📥 Excelでダウンロード", data=excel_data, file_name=f"schedule_{year}{month:02d}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.dataframe(style_table(final_df_for_display))
        
        except Exception as e:
            st.error(f'予期せぬエラーが発生しました: {e}')
            st.exception(e)
    else:
        st.warning('職員一覧と希望休一覧の両方のファイルをアップロードしてください。')

# --- バージョン情報フッター ---
st.markdown("---")
st.markdown(f"<div style='text-align: right; color: grey;'>App Version: {APP_VERSION} with 🤖 Gemini</div>", unsafe_allow_html=True)