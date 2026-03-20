import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(page_title="AI 涂层逆向设计平台", layout="wide")

# ==========================================
# 1. 加载 AI 大脑
# ==========================================
@st.cache_resource # 缓存机制，让网页不用每次刷新都重新加载模型
def load_ai():
    model = joblib.load('coating_ai_brain.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

try:
    ai_model, feature_cols = load_ai()
    ai_ready = True
except:
    ai_ready = False

# ==========================================
# 2. 网页 UI 设计：客户提需求区
# ==========================================
st.title("🧠 AI 驱动：超材料涂层逆向配方生成器")
st.markdown("客户只需输入**性能指标**与**工程限制**，AI 将在毫秒内从 100,000 种可能中为您逆向生成最佳涂料配方。")

st.sidebar.header("🎯 第一步：输入目标性能要求")
target_rsol = st.sidebar.slider("要求的最低太阳光反射比 (%)", 80.0, 98.0, 95.0, step=0.5)
target_emis = st.sidebar.slider("要求的最低大气窗口发射率", 0.80, 0.98, 0.90, step=0.01)

st.sidebar.header("🚧 第二步：设置工程与成本限制")
allow_multilayer = st.sidebar.toggle("允许使用多层/F-P谐振腔 (成本高)", value=False)
allow_plasmonic = st.sidebar.toggle("允许掺杂等离激元金属 (易吸热变色)", value=False)
max_thickness = st.sidebar.number_input("允许的最大施工厚度 (μm)", min_value=100, max_value=500, value=300)

# ==========================================
# 3. AI 逆向生成核心引擎
# ==========================================
if st.button("🚀 启动 AI 逆向设计引擎 (点击生成配方)", type="primary"):
    if not ai_ready:
        st.error("找不到 AI 模型文件！请确保 coating_ai_brain.pkl 已上传。")
    else:
        with st.spinner("AI 正在高维物理空间中进行 100,000 次虚拟打板寻优..."):
            # 1. 生成 10万个随机探索配方
            search_size = 100000
            virtual_recipes = pd.DataFrame({
                '原料_填料折射率 (n)': np.random.uniform(1.5, 2.7, search_size),
                '原料_树脂折射率 (n)': np.random.uniform(1.45, 1.55, search_size),
                '配方_干膜孔隙率 (%)': np.random.uniform(0.0, 40.0, search_size),
                '配方_填料粒径 (μm)': np.random.uniform(0.1, 2.5, search_size),
                '配方_总体积分数_fv (%)': np.random.uniform(10.0, 55.0, search_size),
                '配方_干膜厚度 (μm)': np.random.uniform(100.0, max_thickness, search_size), # 引入客户的厚度限制
                '特征_量子带隙_Eg (eV)': np.random.uniform(3.0, 9.0, search_size),
            })

            # 2. 物理池化预处理
            virtual_recipes['特征_背景有效折射率 (nh)'] = virtual_recipes['原料_树脂折射率 (n)'] * (1 - virtual_recipes['配方_干膜孔隙率 (%)']/100) + 1.0 * (virtual_recipes['配方_干膜孔隙率 (%)']/100)
            virtual_recipes['特征_孔隙TIR增强系数'] = 1.0 + np.where(virtual_recipes['配方_干膜孔隙率 (%)'] > 15.0, (virtual_recipes['配方_干膜孔隙率 (%)']/100 - 0.15) * 1.5, 0)
            
            delta_n = np.maximum(0.01, virtual_recipes['原料_填料折射率 (n)'] - virtual_recipes['特征_背景有效折射率 (nh)'])
            optimal_size = 0.5 / (2 * delta_n)
            size_dev = np.abs(virtual_recipes['配方_填料粒径 (μm)'] - optimal_size) / optimal_size
            size_eff = np.maximum(0.2, 1.0 - (size_dev * 0.6))
            virtual_recipes['特征_微观_散射驱动力'] = (delta_n * 24.0) * size_eff * virtual_recipes['特征_孔隙TIR增强系数']

            # 3. 强加客户的施工约束条件
            virtual_recipes['结构_多层及F-P谐振腔'] = 1 if allow_multilayer else 0
            virtual_recipes['结构_表面渐变折射率'] = np.random.choice([0, 1], search_size)
            virtual_recipes['结构_等离激元金属掺杂'] = 1 if allow_plasmonic else 0

            # 确保列顺序一致
            virtual_recipes = virtual_recipes[feature_cols]

            # 4. AI 毫秒级预测
            predictions = ai_model.predict(virtual_recipes)
            virtual_recipes['预测_反射比'] = predictions[:, 0]
            virtual_recipes['预测_发射率'] = predictions[:, 1]

            # 5. 筛选天选配方
            perfect_recipes = virtual_recipes[
                (virtual_recipes['预测_反射比'] >= target_rsol) & 
                (virtual_recipes['预测_发射率'] >= target_emis)
            ].sort_values(by='预测_反射比', ascending=False)

            # ==========================================
            # 4. 结果展示区
            # ==========================================
            time.sleep(0.5) # 稍微停顿一下，增加 AI 运算的高级感
            st.success(f"寻优完成！在 100,000 个探索方案中，共找到 {len(perfect_recipes)} 个满足所有苛刻条件的配方。")

            if len(perfect_recipes) > 0:
                top_3 = perfect_recipes.head(3).reset_index(drop=True)
                
                # 用漂亮的分栏卡片展示前三名
                cols = st.columns(3)
                for i in range(len(top_3)):
                    with cols[i]:
                        st.markdown(f"### 🏆 推荐方案 {i+1}")
                        st.metric("预估太阳光反射比", f"{top_3.loc[i, '预测_反射比']:.2f} %")
                        st.metric("预估热发射率", f"{top_3.loc[i, '预测_发射率']:.3f}")
                        
                        st.markdown("#### 核心配方参数：")
                        st.info(f"**填料折射率**: {top_3.loc[i, '原料_填料折射率 (n)']:.2f} \n\n"
                                f"**推荐粒径**: {top_3.loc[i, '配方_填料粒径 (μm)']:.2f} μm \n\n"
                                f"**粉体体积浓度(fv)**: {top_3.loc[i, '配方_总体积分数_fv (%)']:.1f} % \n\n"
                                f"**干膜孔隙率**: {top_3.loc[i, '配方_干膜孔隙率 (%)']:.1f} % \n\n"
                                f"**推荐涂装厚度**: {top_3.loc[i, '配方_干膜厚度 (μm)']:.0f} μm")
            else:
                st.warning("⚠️ 您的要求过于严苛（触碰了当前材料系统的物理极限）。建议：降低反射比要求，或者在左侧侧边栏允许使用【多层/F-P谐振腔】结构。")