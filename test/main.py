import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用SimHei，Mac用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False

# 1. 从Excel读取数据
try:
    df = pd.read_excel('data.xlsx')  # 如果文件不在同一目录，需提供完整路径
    print("数据读取成功！前5行数据：")
    print(df.head())
except Exception as e:
    print("读取文件出错:", e)
    exit()

# 2. 统计计算
gender_counts = df['性别'].value_counts()
subject_counts = df['任教学科'].value_counts()
gender_subject = pd.crosstab(df['任教学科'], df['性别'])

# 3. 创建画布
plt.figure(figsize=(15, 10))

# 4. 饼图 - 男女比例
plt.subplot(2, 2, 1)
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('男女比例(饼图)')

# 5. 柱状图 - 男女比例
plt.subplot(2, 2, 2)
gender_counts.plot(kind='bar', color=['pink', 'lightblue'])
plt.title('男女比例(柱状图)')
plt.ylabel('人数')
for i, v in enumerate(gender_counts):
    plt.text(i, v, str(v), ha='center', va='bottom')

# 6. 饼图 - 学科分布
plt.subplot(2, 2, 3)
plt.pie(subject_counts, labels=subject_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('学科分布(饼图)')

# 7. 柱状图 - 学科分布
plt.subplot(2, 2, 4)
subject_counts.plot(kind='bar', color='lightgreen')
plt.title('学科分布(柱状图)')
plt.ylabel('人数')
for i, v in enumerate(subject_counts):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 8. 学科性别分布柱状图
plt.figure(figsize=(10, 6))
gender_subject.plot(kind='bar', color=['pink', 'lightblue'])
plt.title('各学科性别分布')
plt.ylabel('人数')
plt.xlabel('学科')
for i in range(len(gender_subject)):
    for j, col in enumerate(gender_subject.columns):
        count = gender_subject.iloc[i, j]
        if count > 0:  # 只显示有值的标签
            plt.text(i - 0.1 + j*0.2, count, str(count), ha='center')
plt.tight_layout()
plt.show()

# 9. 打印统计信息
print("\n=== 统计信息 ===")
print("\n1. 男女比例:")
print(gender_counts)
print("\n2. 学科分布:")
print(subject_counts)
print("\n3. 各学科性别分布:")
print(gender_subject)

# 10. 保存统计结果
try:
    with pd.ExcelWriter('teacher_stats.xlsx') as writer:
        gender_counts.to_excel(writer, sheet_name='男女比例')
        subject_counts.to_excel(writer, sheet_name='学科分布')
        gender_subject.to_excel(writer, sheet_name='学科性别分布')
    print("\n统计结果已保存到 teacher_stats.xlsx")
except Exception as e:
    print("保存统计结果出错:", e)