import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import calendar
import io
from datetime import datetime

# --- ヘルパー関数: サマリー作成 ---
def _create_summary(schedule_df, staff_info_dict, year, month, event_units):
    num_days = calendar.monthrange(year, month)[1]
    days = list(range(1, num_days + 1))
    daily_summary = []
    for d in days:
        day_info = {}
        work_staff_ids = schedule_df[schedule_df[d] == '出']['職員番号']
        day_info['日'] = d
        day_info['曜日'] = ['月','火','水','木','金','土','日'][calendar.weekday(year, month, d)]
        day_info['出勤者総数'] = len(work_staff_ids)
        day_info['PT'] = sum(1 for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '理学療法士')
        day_info['OT'] = sum(1 for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '作業療法士')
        day_info['ST'] = sum(1 for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '言語聴覚士')
        day_info['役職者'] = sum(1 for sid in work_staff_ids if pd.notna(staff_info_dict[sid]['役職']))
        day_info['回復期'] = sum(1 for sid in work_staff_ids if staff_info_dict[sid].get('役割1') == '回復期専従')
        day_info['地域包括'] = sum(1 for sid in work_staff_ids if staff_info_dict[sid].get('役割1') == '地域包括専従')
        day_info['外来'] = sum(1 for sid in work_staff_ids if staff_info_dict[sid].get('役割1') == '外来PT')
        if calendar.weekday(year, month, d) != 6:
            pt_units = sum(int(staff_info_dict[sid]['1日の単位数']) for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '理学療法士')
            ot_units = sum(int(staff_info_dict[sid]['1日の単位数']) for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '作業療法士')
            st_units = sum(int(staff_info_dict[sid]['1日の単位数']) for sid in work_staff_ids if staff_info_dict[sid]['職種'] == '言語聴覚士')
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
def _create_schedule_df(shifts_values, staff, days, staff_df, requests_x, requests_tri):
    schedule_data = {}
    for s in staff:
        row = []
        s_requests_x = requests_x.get(s, [])
        s_requests_tri = requests_tri.get(s, [])
        for d in days:
            if shifts_values.get((s, d), 0) == 0:
                if d in s_requests_x: row.append('×')
                elif d in s_requests_tri: row.append('△')
                else: row.append('-')
            else: row.append('')
        schedule_data[s] = row
    schedule_df = pd.DataFrame.from_dict(schedule_data, orient='index', columns=days)
    schedule_df = schedule_df.reset_index().rename(columns={'index': '職員番号'})
    staff_map = staff_df.set_index('職員番号')
    schedule_df.insert(1, '職員名', schedule_df['職員番号'].map(staff_map['職員名']))
    schedule_df.insert(2, '職種', schedule_df['職員番号'].map(staff_map['職種']))
    return schedule_df

# --- メインのソルバー関数 (3パターン探索) ---
def solve_three_patterns(staff_df, requests_df, year, month, 
                         target_pt, target_ot, target_st, tolerance,
                         event_units, tri_penalty_weight, min_distance_N):
    # (この関数の中身は変更ありません)
    num_days = calendar.monthrange(year, month)[1]; days = list(range(1, num_days + 1)); staff = staff_df['職員番号'].tolist()
    staff_info = staff_df.set_index('職員番号').to_dict('index')
    sundays = [d for d in days if calendar.weekday(year, month, d) == 6]; weekdays = [d for d in days if d not in sundays]
    managers = [s for s in staff if pd.notna(staff_info[s]['役職'])]; pt_staff = [s for s in staff if staff_info[s]['職種'] == '理学療法士']
    ot_staff = [s for s in staff if staff_info[s]['職種'] == '作業療法士']; st_staff = [s for s in staff if staff_info[s]['職種'] == '言語聴覚士']
    kaifukuki_staff = [s for s in staff if staff_info[s].get('役割1') == '回復期専従']; kaifukuki_pt = [s for s in kaifukuki_staff if staff_info[s]['職種'] == '理学療法士']
    kaifukuki_ot = [s for s in kaifukuki_staff if staff_info[s]['職種'] == '作業療法士']; gairai_staff = [s for s in staff if staff_info[s].get('役割1') == '外来PT']
    chiiki_staff = [s for s in staff if staff_info[s].get('役割1') == '地域包括専従']; sunday_off_staff = gairai_staff + chiiki_staff
    requests_x = {}; requests_tri = {}
    for index, row in requests_df.iterrows():
        staff_id = row['職員番号']; requests_x[staff_id] = [d for d in days if str(d) in requests_df.columns and row.get(str(d)) == '×']; requests_tri[staff_id] = [d for d in days if str(d) in requests_df.columns and row.get(str(d)) == '△']
    def build_model(add_distance_constraint=False, base_solution=None, high_flat_penalty=False):
        model = cp_model.CpModel()
        shifts = {(s, d): model.NewBoolVar(f'shift_{s}_{d}') for s in staff for d in days}
        for s in staff: model.Add(sum(1 - shifts[(s, d)] for d in days) == 9)
        for s, req_days in requests_x.items():
            if s in staff:
                for d in req_days: model.Add(shifts[(s, d)] == 0)
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
            s_requests = requests_x.get(s, []) + requests_tri.get(s, [])
            for w_idx, week in enumerate(weeks_in_month):
                if sum(1 for d in week if d in s_requests) >= 3: continue
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
        unit_penalty_weight = 4 if high_flat_penalty else 2
        staff_penalty_weight = 3 if high_flat_penalty else 1
        job_types = {'PT': pt_staff, 'OT': ot_staff, 'ST': st_staff};
        for job, members in job_types.items():
            if not members: continue
            avg_work_days = (num_days - 9) * len(members); target_per_weekday = avg_work_days / len(weekdays) if weekdays else 0
            for d in weekdays:
                actual = sum(shifts[(s, d)] for s in members); diff = model.NewIntVar(-len(members), len(members), f'd_{job}_{d}'); model.Add(diff == actual - round(target_per_weekday)); abs_diff = model.NewIntVar(0, len(members), f'a_d_{job}_{d}'); model.AddAbsEquality(abs_diff, diff); penalties.append(staff_penalty_weight * abs_diff)
        total_weekday_units = sum(int(staff_info[s]['1日の単位数']) for s in staff) * (len(weekdays)) * (len(staff)-9)/len(staff); total_event_units = sum(event_units.values()); avg_residual_units = (total_weekday_units - total_event_units) / len(weekdays) if weekdays else 0
        for d in weekdays:
            provided_units = sum(shifts[(s, d)] * int(staff_info[s]['1日の単位数']) for s in staff); event_unit = event_units.get(d, 0); residual_units = model.NewIntVar(-2000, 2000, f'r_{d}'); model.Add(residual_units == provided_units - event_unit); diff = model.NewIntVar(-2000, 2000, f'u_d_{d}'); model.Add(diff == residual_units - round(avg_residual_units)); abs_diff = model.NewIntVar(0, 2000, f'a_u_d_{d}'); model.AddAbsEquality(abs_diff, diff); penalties.append(unit_penalty_weight * abs_diff)
        if add_distance_constraint and base_solution:
            diff_vars = []
            for s in staff:
                for d in days:
                    diff_var = model.NewBoolVar(f'diff_{s}_{d}'); model.Add(shifts[(s, d)] != base_solution.get((s, d), 0)).OnlyEnforceIf(diff_var); model.Add(shifts[(s, d)] == base_solution.get((s, d), 0)).OnlyEnforceIf(diff_var.Not()); diff_vars.append(diff_var)
            model.Add(sum(diff_vars) >= min_distance_N)
        model.Minimize(sum(penalties))
        return model, shifts
    
    results = []
    base_solution_values = None
    with st.spinner("パターン1 (最適解) を探索中..."):
        model1, shifts1 = build_model()
        solver1 = cp_model.CpSolver(); solver1.parameters.max_time_in_seconds = 20.0; status1 = solver1.Solve(model1)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return False, [], "致命的エラー: 勤務表を作成できませんでした。ハード制約が矛盾している可能性があります。"
    base_solution_values = {(s, d): solver1.Value(shifts1[(s, d)]) for s in staff for d in days}
    result1 = {"title": "勤務表パターン1", "status": solver1.StatusName(status1), "penalty": round(solver1.ObjectiveValue())}
    result1["schedule_df"] = _create_schedule_df(base_solution_values, staff, days, staff_df, requests_x, requests_tri)
    results.append(result1)
    with st.spinner(f"パターン2 (パターン1と{min_distance_N}マス以上違う解) を探索中..."):
        model2, shifts2 = build_model(add_distance_constraint=True, base_solution=base_solution_values)
        solver2 = cp_model.CpSolver(); solver2.parameters.max_time_in_seconds = 20.0; status2 = solver2.Solve(model2)
    if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution2_values = {(s, d): solver2.Value(shifts2[(s, d)]) for s in staff for d in days}
        result2 = {"title": "勤務表パターン2", "status": solver2.StatusName(status2), "penalty": round(solver2.ObjectiveValue())}
        result2["schedule_df"] = _create_schedule_df(solution2_values, staff, days, staff_df, requests_x, requests_tri)
        results.append(result2)
    with st.spinner("パターン3 (平準化重視) を探索中..."):
        model3, shifts3 = build_model(high_flat_penalty=True)
        solver3 = cp_model.CpSolver(); solver3.parameters.max_time_in_seconds = 20.0; status3 = solver3.Solve(model3)
    if status3 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution3_values = {(s, d): solver3.Value(shifts3[(s, d)]) for s in staff for d in days}
        result3 = {"title": "パターン3 (平準化重視)", "status": solver3.StatusName(status3), "penalty": round(solver3.ObjectiveValue())}
        result3["schedule_df"] = _create_schedule_df(solution3_values, staff, days, staff_df, requests_x, requests_tri)
        results.append(result3)
    return True, results, f"{len(results)}パターンの探索が完了しました。"

def display_result(result_data, staff_info, event_units, year, month):
    st.header(result_data['title'])
    st.info(f"求解ステータス: **{result_data['status']}** (ペナルティ合計: **{result_data['penalty']}**)")
    schedule_df = result_data["schedule_df"]
    temp_work_df = schedule_df.replace({'×': '休', '-': '休', '△': '休', '': '出'})
    summary_df = _create_summary(temp_work_df, staff_info, year, month, event_units)
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
    st.download_button(label=f"📥 {result_data['title']} をExcelでダウンロード", data=excel_data, file_name=f"schedule_{result_data['title']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=result_data['title'])
    st.dataframe(style_table(final_df_for_display))

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title('リハビリテーション科 勤務表作成アプリ')

with st.expander("▼ 各種パラメータを設定する", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("対象年月とファイル")
        # ★★★ 年月選択機能 ★★★
        current_year = datetime.now().year
        year = st.number_input("年（西暦）", min_value=current_year - 5, max_value=current_year + 5, value=current_year)
        month = st.selectbox("月", options=list(range(1, 13)), index=datetime.now().month)
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
        min_distance = st.number_input("パターン2の最低相違マス数(N)", min_value=1, value=50, step=10, help="パターン1と最低でもこれだけ違うマスを持つパターン2を探します。")

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

with st.expander("現在のルール一覧を表示"):
    st.markdown(f"""
    #### 絶対に守るルール（ハード制約）
    - ✅ **H1:** 全員の月間休日数を **9日** にする
    - ✅ **H2:** 希望休 **(×)** を尊重する
    - ✅ **H3:** **役職者** は毎日1人以上出勤する
    - ✅ **H4:** **外来・地域包括** 担当は日曜日に休む
    - ✅ **H5:** 全員、日曜日の出勤は **最大2日** まで

    #### できるだけ守りたいルール（ソフト制約とペナルティ）
    - 🔴 **S0:** **完全な週（7日間）**は **2日以上** 休む (ペナルティ: 200)
    - 🔵 **S1:** **日曜日の出勤人数** を目標値に近づける (ペナルティ: 40～60)
    - 🔵 **S2:** **不完全な週** は **1日以上** 休む (ペナルティ: 25)
    - 🔵 **S3:** **外来担当** が同時に **2人以上** 休むのを避ける (ペナルティ: 10)
    - 🔵 **S4:** **準希望休(△)** を尊重する（現在設定中のペナルティ: **{tri_penalty_weight}**）
    - 🔵 **S5:** **回復期担当** をPT1名, OT1名配置する (ペナルティ: 5)
    - 🔵 **S6:** 平日の **業務負荷（残余単位数）** を平坦にする (ペナルティ: 2)
    - 🔵 **S7:** 平日の **職種ごと人数** を平坦にする (ペナルティ: 1)
    """)

if create_button:
    if staff_file is not None and requests_file is not None:
        try:
            staff_df = pd.read_csv(staff_file); requests_df = pd.read_csv(requests_file)
            if '職員名' not in staff_df.columns:
                staff_df['職員名'] = staff_df['職種'] + " " + staff_df['職員番号'].astype(str)
                st.info("職員一覧に「職員名」列がなかったため、仮の職員名を生成しました。")
            
            is_feasible, results, message = solve_three_patterns(
                staff_df, requests_df, year, month,
                target_pt, target_ot, target_st, tolerance,
                event_units_input, tri_penalty_weight, min_distance
            )
            
            st.success(message)
            if is_feasible:
                staff_info = staff_df.set_index('職員番号').to_dict('index')
                num_results = len(results)
                if num_results > 0:
                    cols = st.columns(num_results)
                    for i, res in enumerate(results):
                        with cols[i]:
                            display_result(res, staff_info, event_units_input, year, month)
        
        except Exception as e:
            st.error(f'予期せぬエラーが発生しました: {e}')
            st.exception(e)
    else:
        st.warning('職員一覧と希望休一覧の両方のファイルをアップロードしてください。')