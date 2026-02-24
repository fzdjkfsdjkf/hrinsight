# app.py
# Streamlit-приложение "HRInsight"
# Требуемые библиотеки: streamlit, pandas, numpy, plotly.express, datetime, calendar
# Запуск: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta, date
import calendar
import io
import base64
import textwrap
import random

# -------------------------
# Настройки страницы
# -------------------------
st.set_page_config(page_title="HRInsight — Аналитика HR", page_icon="👥", layout="wide")

# -------------------------
# Вспомогательные функции
# -------------------------

def svg_logo_html():
    # Простой SVG логотип — встроенный, не требует файлов
    svg = """

    """
    return svg

def parse_dates(df):
    # Преобразовать даты и привести к корректным dtype
    for col in ['hire_date', 'termination_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
    return df

def generate_demo_data(seed=42, n_employees=600, start_year=2022, end_year=2025):
    """
    Генерация демонстрационных кадровых данных с 2022 по 2025 гг.
    Вернет DataFrame со столбцами:
    employee_id, full_name, hire_date, termination_date, department, position, gender, age_at_hire, performance_score
    """
    random.seed(seed)
    np.random.seed(seed)
    departments = ['Продажи', 'Маркетинг', 'ИТ', 'Бухгалтерия', 'HR', 'Производство', 'Логистика', 'Разработка']
    positions = {
        'Продажи': ['Менеджер по продажам', 'Региональный менеджер', 'Аналитик продаж'],
        'Маркетинг': ['Специалист по контенту', 'SMM', 'SEO-специалист'],
        'ИТ': ['Системный администратор', 'DevOps', 'Техподдержка'],
        'Бухгалтерия': ['Бухгалтер', 'Главный бухгалтер'],
        'HR': ['Рекрутер', 'HR-аналитик'],
        'Производство': ['Оператор', 'Мастер участка'],
        'Логистика': ['Координатор', 'Водитель-экспедитор'],
        'Разработка': ['Junior dev', 'Middle dev', 'Senior dev']
    }
    first_names = ['Алексей','Мария','Иван','Анна','Дмитрий','Елена','Сергей','Ольга','Павел','Ирина','Наталья','Кирилл','Татьяна','Виктор']
    last_names = ['Иванов','Петров','Сидоров','Кузнецова','Смирнов','Попова','Васильев','Михайлова','Новиков','Федорова']
    genders = ['М', 'Ж']

    rows = []
    employee_idx = 1000
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)

    # Создаем поток найма: случайные даты найма по всему периоду
    for i in range(n_employees):
        employee_idx += 1
        dept = random.choice(departments)
        pos = random.choice(positions.get(dept, ['Сотрудник']))
        hire_days = (end_date - start_date).days
        hire_date = start_date + timedelta(days=random.randint(0, hire_days))
        # вероятность увольнения: зависит от dept (например, продажи выше)
        base_term_prob = 0.25 if dept in ['Продажи', 'Разработка'] else 0.15
        # модификатор по времени: более свежие наймы меньше вероятно уволены
        years_from_end = (end_date - hire_date).days / 365.0
        term_prob = base_term_prob * (1 - 0.2 * years_from_end)
        terminated = random.random() < term_prob
        if terminated:
            # срок работы до увольнения: 30 дней до 4 лет
            max_days = (end_date - hire_date).days
            if max_days <= 30:
                termination_date = hire_date + timedelta(days=random.randint(1, max_days if max_days>0 else 1))
            else:
                term_days = random.randint(30, min(1460, max_days))
                termination_date = hire_date + timedelta(days=term_days)
                if termination_date > end_date:
                    termination_date = end_date
        else:
            termination_date = pd.NaT

        gender = random.choice(genders)
        full_name = f"{random.choice(last_names)} {random.choice(first_names)}"
        age_at_hire = random.randint(20, 55)
        performance_score = round(np.clip(np.random.normal(3.5, 0.9), 1.0, 5.0), 2)
        rows.append({
            'employee_id': employee_idx,
            'full_name': full_name,
            'hire_date': hire_date,
            'termination_date': termination_date if pd.notnull(termination_date) else pd.NaT,
            'department': dept,
            'position': pos,
            'gender': gender,
            'age_at_hire': age_at_hire,
            'performance_score': performance_score
        })
    df = pd.DataFrame(rows)
    # Некоторая постобработка: сделать даты python date
    df['hire_date'] = pd.to_datetime(df['hire_date']).dt.date
    df['termination_date'] = pd.to_datetime(df['termination_date']).dt.date
    return df

def compute_headcount_timeseries(df, start_date, end_date, freq='M'):
    """
    Возвращает DataFrame с датой и headcount (число сотрудников) на каждую точку времени.
    freq поддерживает 'M' (месяц), 'Q' (квартал)
    """
    # создаем список периодов (последний день месяца)
    idx = pd.date_range(start=start_date, end=end_date, freq='M' if freq == 'M' else 'Q')
    data = []
    for ts in idx:
        d = ts.date()
        # сотрудники, у которых hire_date <= d и (termination_date is null or termination_date > d)
        count = ((df['hire_date'] <= d) & ((df['termination_date'].isna()) | (df['termination_date'] > d))).sum()
        data.append({'period_end': d, 'headcount': int(count)})
    return pd.DataFrame(data)

def month_year(d):
    return d.strftime("%Y-%m")

def calc_turnover(df, period_start, period_end):
    """
    Рассчитывает общую текучесть за период:
    текучесть = увольнения в период / средний headcount * 100
    """
    # увольнения в период (termination_date within)
    term_mask = df['termination_date'].notna() & (df['termination_date'] >= period_start) & (df['termination_date'] <= period_end)
    term_count = term_mask.sum()
    # headcount at start and end
    hc_start = ((df['hire_date'] <= period_start) & ((df['termination_date'].isna()) | (df['termination_date'] > period_start))).sum()
    hc_end = ((df['hire_date'] <= period_end) & ((df['termination_date'].isna()) | (df['termination_date'] > period_end))).sum()
    avg_hc = max((hc_start + hc_end) / 2, 1)  # чтобы не делить на 0
    turnover = term_count / avg_hc * 100
    return turnover, term_count, avg_hc, hc_start, hc_end

def avg_tenure_months(df, only_terminated=False, period_end=None):
    """
    Средний срок работы в месяцах. По умолчанию для всех уволенных (если only_terminated).
    Если period_end задан, то учитываем рабочие периоды до period_end для текущих сотрудников.
    """
    tenures = []
    for _, row in df.iterrows():
        hire = row['hire_date']
        term = row['termination_date']
        if pd.isna(hire):
            continue
        if pd.notna(term):
            if only_terminated:
                length = (pd.to_datetime(term) - pd.to_datetime(hire)).days / 30.44
                tenures.append(length)
            else:
                length = (pd.to_datetime(term) - pd.to_datetime(hire)).days / 30.44
                tenures.append(length)
        else:
            if period_end is not None:
                length = (pd.to_datetime(period_end) - pd.to_datetime(hire)).days / 30.44
                tenures.append(length)
            else:
                # текущие — считаем до сегодня
                length = (pd.to_datetime(date.today()) - pd.to_datetime(hire)).days / 30.44
                tenures.append(length)
    if len(tenures) == 0:
        return 0.0
    return float(np.mean(tenures))

def retention_rate(df, year=1):
    """
    Коэффициент удержания: доля сотрудников, нанятых в некотором году, которые остались через year год(а).
    Реализуем как: (число тех, чей termination_date либо пуст, либо >= hire_date + year*365) / число hires
    """
    hires = df[df['hire_date'].notna()]
    rates = {}
    # группируем по году найма
    hires['hire_year'] = pd.to_datetime(hires['hire_date']).dt.year
    for y in sorted(hires['hire_year'].unique()):
        subset = hires[hires['hire_year'] == y]
        if subset.shape[0] == 0:
            continue
        survived = 0
        for _, row in subset.iterrows():
            hire = pd.to_datetime(row['hire_date'])
            term = row['termination_date']
            cutoff = hire + pd.DateOffset(years=year)
            if pd.isna(term):
                # ещё в компании -> считаем выжившим
                survived += 1
            else:
                if pd.to_datetime(term) >= cutoff:
                    survived += 1
        rates[y] = survived / len(subset) * 100
    return rates

def hires_and_terminations_timeseries(df, start_date, end_date, freq='M'):
    """
    Возвращает DataFrame с колонками period_end, hires, terminations
    """
    idx = pd.date_range(start=start_date, end=end_date, freq='M' if freq == 'M' else 'Q')
    data = []
    for ts in idx:
        d = ts.date()
        # hires in month
        first_day = ts.replace(day=1).date()
        last_day = ts.date()
        hires = ((df['hire_date'] >= first_day) & (df['hire_date'] <= last_day)).sum()
        terms = (df['termination_date'].notna() & (df['termination_date'] >= first_day) & (df['termination_date'] <= last_day)).sum()
        data.append({'period_end': d, 'hires': int(hires), 'terminations': int(terms)})
    return pd.DataFrame(data)

def churn_by_department(df, period_start, period_end):
    """
    Рассчитать текучесть по отделам: увольнения в период / средний headcount отдела
    """
    depts = df['department'].fillna('Не указано').unique()
    rows = []
    for d in depts:
        sub = df[df['department'] == d]
        turnover, term_count, avg_hc, hc_start, hc_end = calc_turnover(sub, period_start, period_end)
        rows.append({'department': d, 'turnover_pct': turnover, 'terminations': term_count, 'avg_headcount': avg_hc})
    return pd.DataFrame(rows).sort_values('turnover_pct', ascending=False)

def avg_age_at_termination(df, period_start=None, period_end=None):
    """
    Средний возраст сотрудников на момент увольнения (если age_at_hire имеется)
    """
    ages = []
    for _, r in df.iterrows():
        if pd.isna(r['termination_date']) or pd.isna(r['hire_date']) or pd.isna(r.get('age_at_hire', np.nan)):
            continue
        # возраст на момент увольнения = age_at_hire + (termination_date - hire_date)/365
        diff_years = (pd.to_datetime(r['termination_date']) - pd.to_datetime(r['hire_date'])).days / 365.25
        ages.append(r['age_at_hire'] + diff_years)
    if len(ages) == 0:
        return None
    return float(np.mean(ages))

def proportion_less_than_one_year(df, period_start=None, period_end=None):
    """
    Доля сотрудников, проработавших менее 1 года (из всех уволенных или из всех? Возьмем из всех уволенных).
    """
    uvol = df[df['termination_date'].notna()]
    if uvol.shape[0] == 0:
        return 0.0
    count = 0
    for _, r in uvol.iterrows():
        tenure_days = (pd.to_datetime(r['termination_date']) - pd.to_datetime(r['hire_date'])).days
        if tenure_days < 365:
            count += 1
    return count / len(uvol) * 100

def months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

def adaptation_rates(df):
    """
    Процент сотрудников, уволившихся в первые 3, 6, 12 месяцев после найма.
    """
    hires = df[df['hire_date'].notna()]
    results = {'3m': 0.0, '6m': 0.0, '12m': 0.0}
    if hires.shape[0] == 0:
        return results
    for _, r in hires.iterrows():
        hire = pd.to_datetime(r['hire_date'])
        term = r['termination_date']
        if pd.isna(term):
            continue
        term = pd.to_datetime(term)
        days = (term - hire).days
        if days <= 90:
            results['3m'] += 1
        if days <= 180:
            results['6m'] += 1
        if days <= 365:
            results['12m'] += 1
    total = hires.shape[0]
    for k in results:
        results[k] = results[k] / total * 100
    return results

def monthly_heatmap_matrix(df, start_date, end_date):
    """
    Создание матрицы (таблицы) увольнений: строки - отделы, столбцы - месяцы, значения - число увольнений
    """
    # Создаём список месяцев в формате 'YYYY-MM'
    months = pd.date_range(
        start=start_date,
        end=end_date,
        freq='ME'          # 'ME' — новый стандарт вместо устаревшего 'M'
    ).strftime('%Y-%m').tolist()

    depts = sorted(df['department'].fillna('Не указано').unique())

    # Создаём пустую матрицу
    mat = pd.DataFrame(0, index=depts, columns=months)

    for _, r in df.iterrows():
        if pd.isna(r['termination_date']):
            continue

        term = pd.to_datetime(r['termination_date'])
        if term.date() < start_date or term.date() > end_date:
            continue

        # Просто берём строку 'YYYY-MM'
        m = term.strftime('%Y-%m')

        d = r['department'] if pd.notna(r['department']) else 'Не указано'
        if m in mat.columns:          # защита от выхода за границы
            mat.loc[d, m] += 1

    return mat

def detect_red_flags(metrics_summary, churn_by_dept_df, hires_ts, terms_ts):
    """
    Простая логика выявления проблем:
    - общий churn > 25%
    - avg tenure < 18 месяцев
    - топ-отделы с churn > avg + 15 п.п.
    - месяцы с резким ростом увольнений (term > mean + 2*sd)
    Возвращаем список флагов и пояснения
    """
    flags = []
    # общий churn
    if metrics_summary.get('turnover_pct', 0) > 25:
        flags.append({
            'title': 'Высокая текучесть',
            'desc': f"Общая текучесть за период составляет {metrics_summary.get('turnover_pct'):.1f}% — выше порога 25%."
        })
    if metrics_summary.get('avg_tenure_years', 0) < 1.5:
        flags.append({
            'title': 'Короткий средний стаж',
            'desc': f"Средний срок работы составляет {metrics_summary.get('avg_tenure_years'):.2f} года(лет), меньше 1.5 лет."
        })
    # по отделам
    overall = metrics_summary.get('turnover_pct', 0)
    for _, r in churn_by_dept_df.iterrows():
        if r['turnover_pct'] > overall + 15:
            flags.append({
                'title': f"Проблемы в отделе {r['department']}",
                'desc': f"Текучесть в отделе {r['department']} составляет {r['turnover_pct']:.1f}% (в среднем {overall:.1f}%). Рекомендуется фокусная проверка."
            })
    # резкие месяцы
    terms_series = terms_ts['terminations'] if 'terminations' in terms_ts.columns else pd.Series()
    if len(terms_series) > 3:
        mean = terms_series.mean()
        sd = terms_series.std()
        spikes = terms_ts[terms_series > mean + 2 * sd]
        for _, row in spikes.iterrows():
            flags.append({
                'title': 'Резкий рост увольнений',
                'desc': f"В месяце {row['period_end']} зафиксировано {row['terminations']} увольнений (среднее {mean:.1f})."
            })
    return flags

def generate_recommendations(flags, churn_by_dept_df, metrics_summary):
    """
    Генерируем 5-8 рекомендаций на русском языке с обоснованием цифрами.
    """
    recs = []
    # Общие рекомендации
    recs.append({
        'rec': "Провести exit-интервью и системный анализ причин увольнений в подразделениях с наивысшей текучестью.",
        'reason': f"Отделы с наибольшей текучестью: {', '.join(churn_by_dept_df.head(3)['department'].tolist())}."
    })
    recs.append({
        'rec': "Пересмотреть систему мотивации и бонусов для отделов с текучестью значительно выше среднего.",
        'reason': f"Средняя текучесть: {metrics_summary.get('turnover_pct'):.1f}%."
    })
    recs.append({
        'rec': "Усилить адаптацию новых сотрудников (онбординг): менторство, чек-листы, регулярные 1:1 в первые 3 месяца.",
        'reason': f"Процент увольнений в первые 3 месяца: {metrics_summary.get('adaptation_3m'):.1f}%."
    })
    recs.append({
        'rec': "Внедрить мониторинг показателей удержания по годам и KPI для менеджеров по найму.",
        'reason': f"Retention 1-го года (по годам): {', '.join([f'{y}:{r:.1f}%' for y,r in metrics_summary.get('retention_1y', {}).items()])}."
    })
    recs.append({
        'rec': "Провести анализ нагрузки и карьерных возможностей в отделах с низким средним стажем.",
        'reason': f"Средний стаж по компании: {metrics_summary.get('avg_tenure_years'):.2f} лет."
    })
    # дополнительные, основанные на флагах
    for f in flags[:3]:
        recs.append({
            'rec': f"Мера по флагу: {f['title']}. Рекомендуется детальная проверка.",
            'reason': f"{f['desc']}"
        })
    # trim to 8
    return recs[:8]

# -------------------------
# Интерфейс и логика приложения
# -------------------------

# Заголовок с логотипом
cols = st.columns([1])
with cols[0]:
    st.markdown("<h1 style='margin-bottom:0'>HRInsight — система анализа и оптимизации HR-процессов</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:gray;margin-top:0'>Интерактивный аналитический инструмент для HR-специалистов и менеджеров</div>", unsafe_allow_html=True)

st.write("---")

# Сайдбар — фильтры и загрузка
st.sidebar.header("Данные и фильтры")
# Тема/цвет — реализуем переключатель визуально (ограниченная кастомизация)
theme_toggle = st.sidebar.selectbox("Тема интерфейса", options=["Авто (по Streamlit)", "Светлая", "Тёмная"])

# Загрузка CSV
uploaded = st.sidebar.file_uploader("Загрузить CSV с данными сотрудников", type=['csv'], help="Ожидаемые столбцы: employee_id, full_name, hire_date, termination_date, department, position, gender, age_at_hire, performance_score")

# Кнопка генерации демо-данных
if st.sidebar.button("Использовать демо-данные 2022–2025"):
    st.session_state['df_hr'] = generate_demo_data(n_employees=800, start_year=2022, end_year=2025)
    st.sidebar.success("Демо-данные загружены в сессию.")

# Если загружен файл
if uploaded is not None:
    try:
        df_uploaded = pd.read_csv(uploaded)
        df_uploaded = parse_dates(df_uploaded)
        st.session_state['df_hr'] = df_uploaded
        st.sidebar.success("Файл успешно загружен.")
    except Exception as e:
        st.sidebar.error(f"Ошибка при чтении CSV: {e}")

# Если нет данных в session_state — инициализируем пустой DF
if 'df_hr' not in st.session_state:
    st.session_state['df_hr'] = pd.DataFrame(columns=['employee_id', 'full_name', 'hire_date', 'termination_date', 'department', 'position', 'gender', 'age_at_hire', 'performance_score'])

df = st.session_state['df_hr']

# Базовые проверки структуры данных — показываем подсказку, если столбцы отсутствуют
required_cols = ['employee_id', 'full_name', 'hire_date', 'termination_date', 'department', 'position', 'gender', 'age_at_hire', 'performance_score']
missing = [c for c in ['employee_id','full_name','hire_date','department'] if c not in df.columns]
if len(df) == 0:
    st.info("Нет загруженных данных. Загрузите CSV или используйте демо-данные.")
else:
    # Опционально сообщаем о недостающих колонках
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"В данных отсутствуют некоторые ожидаемые столбцы: {missing_cols}. Многие метрики не смогут быть рассчитаны полностью.")

# Фильтры периода
min_date = df['hire_date'].min() if not df['hire_date'].isna().all() else date(2022,1,1)
max_date = df['termination_date'].dropna().max() if df['termination_date'].notna().any() else date.today()
if pd.isna(min_date):
    min_date = date(2022,1,1)
if pd.isna(max_date):
    max_date = date.today()

st.sidebar.subheader("Фильтры отчёта")
period_start = st.sidebar.date_input("Дата начала", value=min_date)
period_end = st.sidebar.date_input("Дата окончания", value=max_date if max_date > period_start else date.today())
if period_end < period_start:
    st.sidebar.error("Дата окончания должна быть позже даты начала.")

# Выбор подразделений
all_depts = sorted(df['department'].dropna().unique().tolist()) if 'department' in df.columns else []
selected_depts = st.sidebar.multiselect("Подразделения (фильтр)", options=all_depts, default=all_depts)

# возрастные группы: пользователь может выбрать min/max возраста на момент найма
age_min = int(df['age_at_hire'].min()) if 'age_at_hire' in df.columns and df['age_at_hire'].notna().any() else 20
age_max = int(df['age_at_hire'].max()) if 'age_at_hire' in df.columns and df['age_at_hire'].notna().any() else 60
age_range = st.sidebar.slider("Возраст при приёме (мин/макс)", min_value=18, max_value=70, value=(age_min, age_max))

# Кнопка обновления/применения фильтров
if st.sidebar.button("Применить фильтры и пересчитать"):
    st.rerun()

# -------------------------
# Навигация: вкладки основного интерфейса
# -------------------------
tabs = st.tabs(["Главная / Дашборд", "Загрузка данных", "Обзор данных", "Ключевые HR-метрики", "Анализ по подразделениям и периодам", "Проблемные зоны и риски", "Рекомендации для руководства"])

# Применяем фильтры к df для дальнейших расчётов
def filtered_df(df):
    if df is None or df.empty:
        return df
    res = df.copy()
    # date filters: включаем сотрудников, hire_date <= period_end и (termination_date >= period_start or NaT)
    res = res[(res['hire_date'].notna()) & (pd.to_datetime(res['hire_date']).dt.date <= period_end)]
    # Departments
    if selected_depts:
        res = res[res['department'].isin(selected_depts)]
    # Age filter
    if 'age_at_hire' in res.columns:
        res = res[(res['age_at_hire'] >= age_range[0]) & (res['age_at_hire'] <= age_range[1])]
    return res

fdf = filtered_df(df)

# -------------------------
# Главная / Дашборд
# -------------------------
with tabs[0]:
    st.header("Дашборд — краткая сводка")
    if fdf is None or fdf.empty:
        st.info("Нет данных для отображения на дашборде.")
    else:
        # Прогресс-бар для визуализации обработки
        pb = st.progress(0)
        # считаем основные метрики
        turnover_pct, term_count, avg_hc, hc_start, hc_end = calc_turnover(fdf, period_start, period_end)
        pb.progress(20)
        avg_tenure_months_all = avg_tenure_months(fdf, only_terminated=False, period_end=period_end)
        pb.progress(40)
        avg_tenure_years = avg_tenure_months_all / 12.0
        adaptation = adaptation_rates(fdf)
        pb.progress(60)
        retention_1y = retention_rate(fdf, year=1)
        retention_2y = retention_rate(fdf, year=2)
        pb.progress(80)
        less_than_one_year = proportion_less_than_one_year(fdf)
        pb.progress(100)

        # Визуальные KPI
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Текучесть за период", f"{turnover_pct:.1f} %", delta=f"{term_count} увольнений")
        k2.metric("Средний срок работы", f"{avg_tenure_years:.2f} года(лет)", delta=f"{avg_tenure_months_all:.1f} мес.")
        k3.metric("Доля ушедших < 1 года", f"{less_than_one_year:.1f} %")
        k4.metric("Уволено в период", f"{int(term_count)} чел.")
        st.write("")

        # Headcount и динамика (линия)
        hires_ts = hires_and_terminations_timeseries(df, period_start, period_end, freq='M')
        head_ts = compute_headcount_timeseries(df, period_start, period_end, freq='M')
        fig = px.line(head_ts, x='period_end', y='headcount', title="Динамика headcount (по месяцам)")
        fig.add_bar(x=hires_ts['period_end'], y=hires_ts['hires'], name='Найм (в мес.)', opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Загрузка данных
# -------------------------
with tabs[1]:
    st.header("Загрузка и проверка данных")
    st.markdown("**Ожидаемые столбцы:** `employee_id, full_name, hire_date, termination_date, department, position, gender, age_at_hire, performance_score`.")
    st.markdown("Можно загрузить свой CSV или использовать демо-данные. Даты должны быть в формате YYYY-MM-DD или другом распознаваемом pandas.")
    st.write("")
    # Показываем превью данных
    if df is None or df.empty:
        st.info("Данных нет. Загрузите CSV или используйте демо-данные.")
    else:
        st.subheader("Превью загруженных данных")
        st.dataframe(df.head(200))
        # Кнопки: скачать образец CSV, скачать текущие данные
        sample_csv = "employee_id,full_name,hire_date,termination_date,department,position,gender,age_at_hire,performance_score\n1001,Иванов Иван,2022-05-12,2023-07-01,Продажи,Менеджер по продажам,М,29,3.8\n"
        st.download_button("Скачать образец CSV", data=sample_csv, file_name="sample_hr.csv", mime="text/csv")
        # Скачать текущие данные
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("Скачать загруженные данные (CSV)", data=csv_bytes, file_name="hr_data_export.csv", mime="text/csv")

# -------------------------
# Обзор данных
# -------------------------
with tabs[2]:
    st.header("Обзор данных")
    if fdf is None or fdf.empty:
        st.info("Нет данных для обзора.")
    else:
        st.subheader("Общая структура")
        st.write(f"Количество записей: {len(fdf)}")
        st.write(f"Период охвата: {period_start} — {period_end}")
        # Колонки и пропуски
        st.subheader("Отсутствующие значения по столбцам")
        miss = fdf.isna().sum().reset_index().rename(columns={'index': 'column', 0:'missing'})
        st.dataframe(miss)
        # Распределение по отделам
        st.subheader("Распределение по подразделениям (Top 10)")
        dept_counts = (
            fdf['department']
            .value_counts()
            .rename_axis('department')
            .reset_index(name='count')
        )
        st.dataframe(dept_counts.head(10))
        fig = px.pie(dept_counts, names='department', values='count', title='Распределение сотрудников по отделам')
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Ключевые HR-метрики
# -------------------------
with tabs[3]:
    st.header("Ключевые HR-метрики")
    if fdf is None or fdf.empty:
        st.info("Нет данных для расчёта метрик.")
    else:
        with st.spinner("Рассчитываем метрики..."):
            pb2 = st.progress(0)
            turnover_pct, term_count, avg_hc, hc_start, hc_end = calc_turnover(fdf, period_start, period_end)
            pb2.progress(15)
            avg_tenure_m = avg_tenure_months(fdf, only_terminated=False, period_end=period_end)
            pb2.progress(30)
            avg_tenure_y = avg_tenure_m / 12.0
            adaptation_stats = adaptation_rates(fdf)
            pb2.progress(45)
            retention1 = retention_rate(fdf, year=1)
            retention2 = retention_rate(fdf, year=2)
            pb2.progress(60)
            hires_ts = hires_and_terminations_timeseries(fdf, period_start, period_end)
            pb2.progress(80)
            avg_age_term = avg_age_at_termination(fdf)
            less_1y = proportion_less_than_one_year(fdf)
            pb2.progress(100)

        st.subheader("Сводка по основным метрикам")
        st.metric("Текучесть за период", f"{turnover_pct:.1f}%")
        st.metric("Средний срок работы", f"{avg_tenure_y:.2f} года(лет) / {avg_tenure_m:.1f} мес.")
        st.metric("Доля уволенных <1 года", f"{less_1y:.1f}%")
        st.write(f"Средний возраст на момент увольнения: {avg_age_term:.1f} лет" if avg_age_term is not None else "Недостаточно данных для расчёта среднего возраста при увольнении.")

        # Динамика найма и увольнений (линия)
        st.subheader("Динамика найма и увольнений по месяцам")
        ts = hires_and_terminations_timeseries(fdf, period_start, period_end, freq='M')
        fig = px.line(ts, x='period_end', y=['hires','terminations'], labels={'value':'Число', 'period_end':'Период'}, title="Найм и увольнения")
        st.plotly_chart(fig, use_container_width=True)

        # Retention по годам (таблица)
        st.subheader("Retention rate (удержание) по годам найма")
        r1 = retention1
        if r1:
            r1_df = pd.DataFrame({'hire_year': list(r1.keys()), 'retention_1y_%': list(r1.values())})
            st.dataframe(r1_df)
        else:
            st.write("Недостаточно данных для расчёта retention по годам.")

# -------------------------
# Анализ по подразделениям и периодам
# -------------------------
with tabs[4]:
    st.header("Анализ по подразделениям и периодам")
    if fdf is None or fdf.empty:
        st.info("Нет данных для анализа.")
    else:
        # Текучесть по отделам (bar)
        st.subheader("Текучесть по отделам (за выбранный период)")
        churn_by_dept_df = churn_by_department(fdf, period_start, period_end)
        st.dataframe(churn_by_dept_df)
        fig = px.bar(churn_by_dept_df, x='department', y='turnover_pct', title='Текучесть по отделам (%)', labels={'turnover_pct':'Текучесть %','department':'Отдел'})
        st.plotly_chart(fig, use_container_width=True)

        # Top-5 проблемных должностей (по текучести или числу увольнений)
        st.subheader("Top-5 должностей по числу увольнений")
        if 'position' in fdf.columns:
            pos_terms = (
                fdf[fdf['termination_date'].notna()]['position']
                .value_counts()
                .rename_axis('position')
                .reset_index(name='terminations')
            )
            st.dataframe(pos_terms.head(10))
        else:
            st.write("Столбец 'position' отсутствует в данных.")

        # Boxplot: длительность работы по отделам
        st.subheader("Box-plot длительности работы (в месяцах) по отделам")
        # Считаем tenure для уволенных по отделам
        box_df = []
        for _, r in fdf.iterrows():
            if pd.isna(r['hire_date']):
                continue
            hire = pd.to_datetime(r['hire_date'])
            term = r['termination_date']
            if pd.notna(term):
                term = pd.to_datetime(term)
            else:
                term = pd.to_datetime(period_end)
            tenure_m = (term - hire).days / 30.44
            box_df.append({'department': r['department'] if pd.notna(r['department']) else 'Не указано', 'tenure_m': tenure_m})
        box_df = pd.DataFrame(box_df)
        if not box_df.empty:
            fig = px.box(box_df, x='department', y='tenure_m', title='Длительность работы по отделам (месяцы)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Недостаточно данных для построения box-plot.")

        # Heatmap увольнений по месяцам и отделам
        st.subheader("Heatmap увольнений (отделы × месяцы)")
        heat_mat = monthly_heatmap_matrix(fdf, period_start, period_end)
        st.dataframe(heat_mat)
        try:
            fig = px.imshow(heat_mat.values, labels=dict(x="Месяц", y="Отдел", color="Увольнения"),
                            x=heat_mat.columns, y=heat_mat.index, title="Heatmap увольнений")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.write("Невозможно отобразить heatmap графически — покажем таблицу.")

# -------------------------
# Проблемные зоны и риски
# -------------------------
with tabs[5]:
    st.header("Проблемные зоны и риски")
    if fdf is None or fdf.empty:
        st.info("Нет данных для выявления рисков.")
    else:
        # Сводка метрик для детекции
        metrics_summary = {}
        turnover_pct, term_count, avg_hc, hc_start, hc_end = calc_turnover(fdf, period_start, period_end)
        metrics_summary['turnover_pct'] = turnover_pct
        avg_tenure_m = avg_tenure_months(fdf, only_terminated=False, period_end=period_end)
        metrics_summary['avg_tenure_months'] = avg_tenure_m
        metrics_summary['avg_tenure_years'] = avg_tenure_m / 12.0
        metrics_summary['adaptation_3m'] = adaptation_stats.get('3m', 0.0) if 'adaptation_stats' in locals() else adaptation_rates(fdf)['3m']
        metrics_summary['retention_1y'] = retention1 if 'retention1' in locals() else retention_rate(fdf,1)
        # timeseries for spikes
        terms_ts = hires_and_terminations_timeseries(fdf, period_start, period_end, freq='M')
        hires_ts = terms_ts.copy()
        flags = detect_red_flags(metrics_summary, churn_by_dept_df, hires_ts, terms_ts)

        if len(flags) == 0:
            st.success("Явных красных флагов не обнаружено по текущим правилам.")
        else:
            st.warning("Обнаружены потенциальные проблемные зоны:")
            for f in flags:
                st.markdown(f"**{f['title']}** — {f['desc']}")

        st.subheader("Риски по отделам (топ-5 по текучести)")
        st.dataframe(churn_by_dept_df.head(10))

# -------------------------
# Рекомендации для руководства
# -------------------------
with tabs[6]:
    st.header("Рекомендации для руководства")
    if fdf is None or fdf.empty:
        st.info("Нет данных для генерации рекомендаций.")
    else:
        # Генерируем рекомендации
        recs = generate_recommendations(flags, churn_by_dept_df, metrics_summary)
        st.subheader("Автоматические рекомендации (5–8 пунктов)")
        rec_texts = []
        for i, r in enumerate(recs, start=1):
            st.markdown(f"**{i}. {r['rec']}**")
            st.caption(r['reason'])
            rec_texts.append(f"{i}. {r['rec']} — {r['reason']}")
        # Кнопка копирования в буфер обмена (через JS)
        rec_plain = "\n".join(rec_texts)
        copy_button_html = f"""
        <button onclick="navigator.clipboard.writeText(`{rec_plain}`)" style="background-color:#0f4c81;color:white;padding:8px 12px;border-radius:6px;border:none;">Скопировать рекомендации в буфер</button>
        """
        st.markdown(copy_button_html, unsafe_allow_html=True)
        st.write("")


# -------------------------
# Небольшой footer и подсказки
# -------------------------
st.write("---")
cols2 = st.columns([1])
with cols2[0]:
    st.write("© HRInsight — аналитическая платформа.")
