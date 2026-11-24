# 同步工具通过线程池放到事件循环之外执行；异步包装本身不保证多个协程已同时启动。
# 当前快捷函数的 async with 与执行器的同步上下文接口不匹配；这里标注边界，不改变原行为。

"""异步工具执行器 - HelloAgents异步工具执行支持"""

import asyncio
import concurrent.futures
from typing import Dict, Any, List
from .registry import ToolRegistry


class AsyncToolExecutor:
    """异步工具执行器"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        # 学习提示：线程池在对象创建时准备，使用后需 close/shutdown；max_workers 是线程数上限。
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """异步执行单个工具"""
        # 学习提示：通过 run_in_executor 调用同步注册表，避免工具函数直接占用事件循环线程。
        loop = asyncio.get_event_loop()
        
        def _execute():
            return self.registry.execute_tool(tool_name, input_data)
        
        try:
            # 注册表是同步接口，放到线程池执行可避免直接阻塞事件循环。
            result = await loop.run_in_executor(self.executor, _execute)
            return result
        except Exception as e:
            return f"❌ 工具 '{tool_name}' 异步执行失败: {e}"

    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        并行执行多个工具
        
        Args:
            tasks: 任务列表，每个任务包含 tool_name 和 input_data
            
        Returns:
            执行结果列表，包含任务信息和结果
        """
        # 学习提示：当前先保存协程对象再逐个 await，未 create_task/gather，因此此方法目前依次执行。
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")
        
        # 创建异步任务
        async_tasks = []
        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            input_data = task.get("input_data", "")
            
            if not tool_name:
                continue
                
            print(f"📝 创建任务 {i+1}: {tool_name}")
            # 此处只创建协程对象，并未通过 create_task 或 gather 同时调度。
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append((i, task, async_task))
        
        # 等待所有任务完成
        results = []
        for i, task, async_task in async_tasks:
            try:
                # 逐个 await 使当前实现依次执行；函数名中的 parallel 不代表已并发。
                result = await async_task
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": result,
                    # 此状态仅表示 await 正常返回；result 仍可能是注册表生成的错误文本。
                    "status": "success"
                })
                print(f"✅ 任务 {i+1} 完成: {task['tool_name']}")
            except Exception as e:
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": str(e),
                    "status": "error"
                })
                print(f"❌ 任务 {i+1} 失败: {task['tool_name']} - {e}")
        
        print(f"🎉 并行执行完成，成功: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
        return results

    async def execute_tools_batch(self, tool_name: str, input_list: List[str]) -> List[Dict[str, Any]]:
        """
        批量执行同一个工具
        
        Args:
            tool_name: 工具名称
            input_list: 输入数据列表
            
        Returns:
            执行结果列表
        """
        # 学习提示：把一个工具的多个输入转换为任务列表，结果和错误语义继承 execute_tools_parallel。
        tasks = [
            {"tool_name": tool_name, "input_data": input_data}
            for input_data in input_list
        ]
        return await self.execute_tools_parallel(tasks)

    def close(self):
        """关闭执行器"""
        # 学习提示：shutdown(wait=True) 等待线程完成；不会自动取消正在执行的工具动作。
        self.executor.shutdown(wait=True)
        print("🔒 异步工具执行器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
async def run_parallel_tools(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函数：并行执行多个工具
    
    Args:
        registry: 工具注册表
        tasks: 任务列表
        max_workers: 最大工作线程数
        
    Returns:
        执行结果列表
    """
    # 学习提示：当前 async with 要求 __aenter__/__aexit__，但类只实现同步接口，调用会在执行任务前失败。
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_parallel(tasks)


async def run_batch_tool(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    便捷函数：批量执行同一个工具
    
    Args:
        registry: 工具注册表
        tool_name: 工具名称
        input_list: 输入数据列表
        max_workers: 最大工作线程数
        
    Returns:
        执行结果列表
    """
    # 学习提示：与并行快捷入口有相同上下文管理接口不匹配，不能因签名是 async 就认为已可运行。
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_batch(tool_name, input_list)


# 同步包装函数（为了兼容性）
def run_parallel_tools_sync(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本的并行工具执行"""
    # 学习提示：asyncio.run 适用于无运行中事件循环的同步入口，在已有异步循环内直接调用会失败。
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers))


def run_batch_tool_sync(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本的批量工具执行"""
    # 学习提示：仅包装异步快捷函数，不修复其 async with 资源管理问题。
    return asyncio.run(run_batch_tool(registry, tool_name, input_list, max_workers))


# 示例函数
async def demo_parallel_execution():
    """演示并行执行的示例"""
    from .registry import ToolRegistry
    
    # 创建注册表（这里假设已经注册了工具）
    registry = ToolRegistry()
    
    # 定义并行任务
    tasks = [
        {"tool_name": "my_calculator", "input_data": "2 + 2"},
        {"tool_name": "my_calculator", "input_data": "3 * 4"},
        {"tool_name": "my_calculator", "input_data": "sqrt(16)"},
        {"tool_name": "my_calculator", "input_data": "10 / 2"},
    ]
    
    # 并行执行
    results = await run_parallel_tools(registry, tasks)
    
    # 显示结果
    print("\n📊 并行执行结果:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['tool_name']}({result['input_data']}) = {result['result']}")
    
    return results


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_parallel_execution())
