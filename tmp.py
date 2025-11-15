import streamlit as st
import pandas as pd
import random as r
import altair as alt
import numpy as np
from datetime import datetime,timedelta
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency
alt.renderers.set_embed_options(renderer="svg")

BASE_DIR = Path().parent
DATA_DIR = BASE_DIR / "data"
CSV_contact_PATH = DATA_DIR / "records_contact.csv"
CSV_tran_PATH = DATA_DIR / "records_transaction.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

tab1,tab2,tab3=st.tabs(["管道与成交分析","RFM 分析","A/B Test 分析"])

def stop(df):
    if df is None:
        st.error("请上传文件")
        st.stop()

def random_contact():
    return pd.DataFrame({"id":list(range(1,101)),
                         "name":[f"客户{i}" for i in range(100)],
                         "company":sorted([f"公司{r.randint(1,50)}" for _ in range(100)]),
                         "stage":[r.choice(["Qualified","Proposal","Lost","Lead","Won"])for _ in range(100)],
                         "Deal_value":[r.randint(6000,200000) for _ in range(100)]})
def random_transaction(df_contact):
    a=datetime(2025,1,1)-datetime(2005,1,1)
    return pd.DataFrame({"customer_id":[r.choice(range(1,101)) for _ in range(10000)],
                            "date":[datetime(2005,1,1)+timedelta(days=r.randint(0,a.days)) for _ in range(10000)],
                            "amount":[r.randint(6000,200000) for _ in range(10000)]},
                            )


with st.sidebar:
    st.subheader("基础数据生成与上传")
    demo=st.toggle("选择使用随机数据",value=True)
    if demo:
        df_contact=random_contact() if not CSV_contact_PATH.exists() else pd.read_csv(CSV_contact_PATH) 
        df_contact.to_csv(CSV_contact_PATH,index=False)
        
        df_tran=random_transaction(df_contact) if not CSV_tran_PATH.exists() else pd.read_csv(CSV_tran_PATH)
        df_tran.to_csv(CSV_tran_PATH,index=False)
        with st.form("刷新数据"):
            button=st.form_submit_button("刷新数据")
            if button:
                df_contact=random_contact() 
                df_contact.to_csv(CSV_contact_PATH,index=False)
                
                df_tran=random_transaction(df_contact) 
                df_tran.to_csv(CSV_tran_PATH,index=False)
                st.success("数据刷新完成")
    else:
        up_A = st.file_uploader("上传 CSV（列包含：name, company,stage,Deal_value）", type=["csv"])
        up_B = st.file_uploader("上传 CSV（列包含：customer_id,date,amount）", type=["csv"])
        df_contact=pd.read_csv(up_A) if up_A else None
        df_tran=pd.read_csv(up_B) if up_B else None
        if not up_A and not up_B:
            st.error("请上传文件")
            st.stop()
    if st.checkbox("显示数据"):
        st.write(df_contact)
        st.write(df_tran)


with tab1:
    stop(df_contact)
    g=df_contact.groupby("stage")
    cnt=g["id"].count().rename("leads")
    val=g["Deal_value"].sum().rename("values")
    stage_count=pd.concat([cnt,val],axis=1).reset_index()
    bara=alt.Chart(stage_count).mark_bar()
    chart1=bara.encode(x="stage:N",y="leads:Q")
    chart2=bara.encode(x="stage:N",y="values:Q")
    st.altair_chart(chart1,use_container_width=True)
    st.altair_chart(chart2,use_container_width=True)

    # 计算成交和流失数量
    won = (df_contact["stage"] == "Won").sum()
    lost = (df_contact["stage"] == "Lost").sum()
    total = won + lost
    won_rate = won / total if total > 0 else 0.0
    
    # 创建指标展示列
    cols = st.columns(4)
    metrics = [
        ("Won 数", f"{won:.0f}"),
        ("Lost 数", f"{lost:.0f}"),
        ("Total 数", f"{total:.0f}"),
        ("Won 率", f"{won_rate:.2%}")
    ]
    
    # 批量创建指标
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
        
    if st.checkbox("显示数据"):
        st.write(stage_count)

with tab2:
    stop(df_tran)
    df_tran["date"]=pd.to_datetime(df_tran["date"])
    points=st.slider("选择分位数",min_value=2,max_value=100,value=5)
    df_tran_cus=df_tran.groupby("customer_id")
    today=df_tran_cus.date.max()+pd.Timedelta(days=1)
    last_date=df_tran_cus.date.max().rename("last_date")
    freq=df_tran_cus.date.count().rename("freq")
    sum_amount=df_tran_cus.amount.sum().rename("sum_amount")
    
    rfm_df = pd.concat([last_date, freq, sum_amount], axis=1).reset_index()
    rfm_df['r_score'] = pd.qcut(rfm_df['last_date'], q=points, labels=False, duplicates='drop') + 1
    rfm_df['f_score'] = pd.qcut(rfm_df['freq'], q=points, labels=False, duplicates='drop') + 1
    rfm_df['m_score'] = pd.qcut(rfm_df['sum_amount'], q=points, labels=False, duplicates='drop') + 1
    rfm_df['rfm_score'] = rfm_df['r_score'] + rfm_df['f_score'] + rfm_df['m_score']
    
    chart1=alt.Chart(rfm_df).mark_circle(size=90).encode(
        x="f_score:Q",
        y="m_score:Q",
        color="rfm_score:Q"
    )
    st.altair_chart(chart1,use_container_width=True)
    


with tab3:
    
    stop(df_contact)
    mask = df_contact.stage.isin(["Lead","Qualified","Proposal"])
    
    # 仅在需要时显示原始数据
    with st.expander("查看符合条件的潜在客户"):
        st.write(df_contact[mask])
    
    CSV_conv_PATH = DATA_DIR / "records_conversation.csv"
    
    # 改进表单布局和交互体验
    st.subheader("A/B测试数据管理")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("修改响应数据", clear_on_submit=True):
            st.write("重置现有测试数据")
            button = st.form_submit_button("删除当前响应数据")
        if button and CSV_conv_PATH.exists():
            CSV_conv_PATH.unlink()
            st.success("原响应数据已删除，可以重新开始测试")
    
    with col2:
        st.info("💡 提示：删除数据后可以重新分配变体并标记响应情况")
    
    if CSV_conv_PATH.exists():
        st.success("检测到已存在的测试数据，将基于现有数据进行分析")
        df_eligible = pd.read_csv(CSV_conv_PATH)
    else:
        st.info("未检测到测试数据，请为潜在客户分配变体并标记响应情况")
        with st.form("响应数据", clear_on_submit=True):
            # 随机分配变体
            df_contact.loc[mask, "variant"] = np.random.choice(["A","B"], size=mask.sum())
            df_contact.loc[mask, "responded"] = False
            
            st.write("请标记每个客户是否响应了您的营销活动：")
            uploaded_file = st.file_uploader("上传 CSV（列包含：id,variant,responded）(若无则随机分配)", type=["csv"])
            if uploaded_file is not None:
                df_eligible = pd.read_csv(uploaded_file)
            else:
                df_eligible = df_contact[mask][["id","variant","responded"]].sample(frac=r.random())

            submit = st.form_submit_button("提交响应数据", type="primary")
        if submit:
            for i in df_eligible.index:
                df_contact.loc[i,"responded"]=True
            df_eligible=df_contact[mask][["id","variant","responded"]]
            df_eligible.to_csv(CSV_conv_PATH, index=0)
            st.success("响应数据已保存，正在进行分析...")

    # 添加更多A/B测试指标
    st.subheader("详细A/B测试指标")

   # 计算各变体的响应率（用于图表显示）
    df_conv = df_eligible.groupby("variant").responded.mean().reset_index(name="conversation")
    
    # 实现统计显著性检验
    if len(df_eligible) > 0:
        # 创建列联表
        contingency_table = pd.crosstab(df_eligible['variant'], df_eligible['responded'])
        
        # 执行卡方检验
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # 显示检验结果
        st.subheader("统计显著性检验结果")
        st.write(f"卡方统计量: {chi2:.4f}")
        st.write(f"p值: {p_value:.4f}")
        
        # 解释结果
        alpha = 0.05
        if p_value < alpha:
            st.success("结果具有统计显著性 (p < 0.05)，两个变体之间的差异不太可能是偶然的。")
        else:
            st.info("结果不具有统计显著性 (p ≥ 0.05)，两个变体之间的差异可能是偶然的。")
        
        # 显示列联表
        st.subheader("观测数据列联表")
        st.write(contingency_table)
    
    # 修复并完善响应率可视化图表
    st.subheader("A/B变体响应率对比")
    
    # 创建柱状图
    chart = alt.Chart(df_conv).mark_bar(size=100).encode(
        x=alt.X('variant:N', title='变体'),
        y=alt.Y('conversation:Q', 
                title='响应率',
                axis=alt.Axis(format='.1%')),
        color=alt.Color('variant:N', 
                       scale=alt.Scale(range=['#3498db', '#e74c3c']),
                       legend=None),
        tooltip=[
            alt.Tooltip('variant:N', title='变体'),
            alt.Tooltip('conversation:Q', title='响应率', format='.2%')
        ]
    ).properties(
        width=400,
        height=300
    )
    
    # 添加文本标签
    text = chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        text=alt.Text('conversation:Q', format='.1%')
    )
    
    # 组合图表
    final_chart = (chart + text).configure_view(strokeWidth=0)
    
    st.altair_chart(final_chart, use_container_width=True)
    
    
    