import time
import json
from main import graph

# ============ 测试题库 ============

TEST_CASES = [
    "写一个函数计算两个数的最大公约数",
    "写一个函数判断一个数是否是质数",
    "写一个函数实现冒泡排序",
    "写一个函数计算斐波那契数列第n项",
    "写一个函数反转字符串",
    "写一个函数统计字符串中每个字符出现的次数",
    "写一个函数实现二分查找",
    "写一个函数计算列表的中位数",
    "写一个函数判断字符串是否是回文",
    "写一个函数将列表去重并保持原顺序",
]

# ============ 评估函数 ============

def count_repairs(messages: list) -> int:
    """统计自我修复次数（执行报错的次数）"""
    count = 0
    for msg in messages:
        if msg.get("role") == "tool" and "执行报错" in (msg.get("content") or ""):
            count += 1
    return count


def run_eval():
    print("CodePilot 评估开始")
    print("=" * 50)
    print(f"测试题目数量: {len(TEST_CASES)}")
    print("=" * 50)

    results = []
    success_count = 0
    total_repairs = 0
    total_time = 0

    for i, task in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {task}")

        start_time = time.time()
        success = False
        repairs = 0

        try:
            final_state = graph.invoke({
                "user_request": task,
                "plan": "",
                "code_result": "",
                "review": "",
                "messages": [],
                "finished": False
            })

            repairs = count_repairs(final_state.get("messages", []))
            success = True
            success_count += 1

        except Exception as e:
            print(f"  ❌ 异常: {str(e)[:100]}")

        elapsed = round(time.time() - start_time, 2)
        total_time += elapsed
        total_repairs += repairs

        status = "✅" if success else "❌"
        print(f"  {status} 耗时: {elapsed}s | 修复次数: {repairs}")

        results.append({
            "task": task,
            "success": success,
            "repairs": repairs,
            "time": elapsed
        })

    # ============ 统计结果 ============

    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)

    total = len(TEST_CASES)
    success_rate = round(success_count / total * 100, 1)
    avg_repairs = round(total_repairs / total, 2)
    avg_time = round(total_time / total, 2)
    one_shot_count = sum(1 for r in results if r["success"] and r["repairs"] == 0)
    one_shot_rate = round(one_shot_count / total * 100, 1)

    print(f"总题目数:     {total}")
    print(f"成功数:       {success_count}")
    print(f"总成功率:     {success_rate}%")
    print(f"一次成功率:   {one_shot_rate}%  （无需修复直接通过）")
    print(f"平均修复次数: {avg_repairs} 次")
    print(f"平均耗时:     {avg_time} 秒")

    # 保存结果
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "success_count": success_count,
                "success_rate": f"{success_rate}%",
                "one_shot_rate": f"{one_shot_rate}%",
                "avg_repairs": avg_repairs,
                "avg_time_seconds": avg_time
            },
            "details": results
        }, f, ensure_ascii=False, indent=2)

    print("\n详细结果已保存到 eval_results.json")
    print(f"  在 {total} 道标准编程题测试中，任务成功率 {success_rate}%，")
    print(f"  一次成功率 {one_shot_rate}%，平均自我修复 {avg_repairs} 次，平均响应 {avg_time} 秒")


if __name__ == "__main__":
    run_eval()