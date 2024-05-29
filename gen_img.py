import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为新宋体。
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像时 负号'-' 显示为方块的问题。

# 提取红塔红土盛世普益混合发起式在2023年第三季度的境内股票投资组合数据
季度报告_data = {
    'code': [1, 2, 3, 4, 5],
    'industry_category': ['A', 'B', 'C', 'D', 'E'],
    'value': [1000000, 2000000, 3000000, 4000000, 5000000],
    'net_value_ratio': [20, 30, 40, 50, 60]
}

# 计算各行业名称及相应净值占比
industry_names = list(季度报告_data['industry_category'])
net_value_ratios = list(季度报告_data['net_value_ratio'])

# 绘制饼状统计图
plt.pie(net_value_ratios, labels=industry_names, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # 调整图形的比例，使各个行业的比例总和为1
plt.title('红塔红土盛世普益混合发起式证券投资基金2023年第三季度按行业分类的境内股票投资组合')
plt.show()

# # 数据
# quarters = [91.02, 90.81, 91.02, 91.02]
# quarter_names = ['第一季度', '第二季度', '第三季度', '第四季度']
# # 绘制折线图
# plt.figure(figsize=(10, 6))
# plt.plot(quarter_names, quarters , marker='o', linestyle='-')
# plt.title('东方精选混合2023年季度股票投资占比')
# plt.xlabel('季度')
# plt.ylabel('股票投资占比（%）')
# plt.xticks(rotation=90)
# plt.grid(True)
# plt.show()
