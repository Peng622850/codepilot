import time
import json
from datetime import datetime
from main import graph

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


def count_repairs(messages: list) -> int:
    return sum(
        1 for msg in messages
        if msg.get("role") == "tool" and "执行报错" in (msg.get("content") or "")
    )


def is_success(final_state: dict) -> bool:
    """
    真正的成功判定：
    1. code_result 非空
    2. messages 里最后一次工具调用结果不是报错
    3. 没有未处理的异常关键词
    """
    code_result = final_state.get("code_result", "")
    if not code_result:
        return False

    messages = final_state.get("messages", [])
    # 找最后一条 tool 消息
    last_tool_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            last_tool_content = msg.get("content", "")
            break

    # 最后一次执行必须是成功的
    if "执行报错" in last_tool_content:
        return False
    if "执行超时" in last_tool_content:
        return False

    return True


def run_eval():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"eval_results_{timestamp}.json"

    print("CodePilot 评估开始")
    print("=" * 50)
    print(f"测试题目数量: {len(TEST_CASES)}")
    print(f"结果将保存至: {output_file}")
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
            success = is_success(final_state)  # 用严格判定替代 success=True
            if success:
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

    total = len(TEST_CASES)
    success_rate = round(success_count / total * 100, 1)
    avg_repairs = round(total_repairs / total, 2)
    avg_time = round(total_time / total, 2)
    one_shot_count = sum(1 for r in results if r["success"] and r["repairs"] == 0)
    one_shot_rate = round(one_shot_count / total * 100, 1)

    print("\n" + "=" * 50)
    print(f"总成功率:     {success_rate}%")
    print(f"一次成功率:   {one_shot_rate}%")
    print(f"平均修复次数: {avg_repairs} 次")
    print(f"平均耗时:     {avg_time} 秒")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "timestamp": timestamp,
                "total": total
            },
            "summary": {
                "success_count": success_count,
                "success_rate": f"{success_rate}%",
                "one_shot_rate": f"{one_shot_rate}%",
                "avg_repairs": avg_repairs,
                "avg_time_seconds": avg_time
            },
            "details": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到 {output_file}")


if __name__ == "__main__":
    run_eval()