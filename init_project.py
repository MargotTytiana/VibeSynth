import os
from pathlib import Path

# 定义项目目录结构和初始文件内容
PROJECT_STRUCTURE = {
    "vibe-synth": {
        "backend": {
            "api": {
                "__init__.py": "",
                "routes.py": "# 路由定义：处理前端发来的 Vibe 请求\n"
            },
            "core": {
                "__init__.py": "",
                "config.py": "# 环境变量和 LLM API Key 配置\n",
                "llm_client.py": "# 封装与大语言模型（如 GPT-4 / Claude）的交互\n"
            },
            "generators": {
                "__init__.py": "",
                "dsp_generator.py": "# 核心逻辑：将自然语言转换为 Faust/C++ 音频算法\n",
                "shader_generator.py": "# 核心逻辑：将自然语言转换为 GLSL/WGSL 着色器代码\n"
            },
            "compilers": {
                "__init__.py": "",
                "wasm_builder.py": "# 调度底层编译器，将生成的算法代码编译为 WebAssembly\n"
            },
            "prompts": {
                "system_prompt_dsp.txt": "You are an expert DSP engineer. Write Faust code based on the user's vibe description...",
                "system_prompt_shader.txt": "You are a graphics programmer. Write WebGPU shaders..."
            },
            "main.py": (
                "from fastapi import FastAPI\n\n"
                "app = FastAPI(title='Vibe-Synth API')\n\n"
                "@app.get('/')\n"
                "def read_root():\n"
                "    return {'status': 'Latent Space Engine is running'}\n"
            ),
            "requirements.txt": "fastapi\nuvicorn\nopenai\npydantic\n"
        },
        "frontend": {
            "src": {
                "audio": {
                    "vibe_worklet.js": "// 运行在 AudioWorklet 线程中的音频处理器\n",
                    "audio_engine.ts": "// 管理 Web Audio API 上下文和节点连接\n"
                },
                "graphics": {
                    "renderer.ts": "// WebGPU/WebGL 渲染上下文管理\n"
                },
                "components": {
                    "DynamicKnob.tsx": "// 动态生成的旋钮 UI 组件\n",
                    "ControlPanel.tsx": "// 根据后端返回的 JSON 自动渲染的参数控制面板\n"
                },
                "hooks": {
                    "useVibeStream.ts": "// 处理麦克风输入和实时数据流\n"
                },
                "App.tsx": "// 前端主入口\n"
            },
            "package.json": "{\n  \"name\": \"vibe-synth-frontend\",\n  \"version\": \"1.0.0\"\n}\n"
        },
        "shared": {
            "api_contracts.json": "{\n  \"description\": \"前后端交互的 JSON Schema 定义\"\n}\n"
        },
        "README.md": "# Project Vibe-Synth\n\n隐空间算法合成器 (Latent Space DSP/Shader Generator) 核心代码库。\n"
    }
}

def create_structure(base_path: Path, structure: dict):
    """递归创建目录和文件"""
    for name, content in structure.items():
        current_path = base_path / name
        
        if isinstance(content, dict):
            # 如果是字典，说明是文件夹
            print(f"📁 创建目录: {current_path}")
            current_path.mkdir(parents=True, exist_ok=True)
            # 递归处理子目录
            create_structure(current_path, content)
        else:
            # 如果是字符串，说明是文件，创建并写入内容
            print(f"📄 创建文件: {current_path}")
            with open(current_path, "w", encoding="utf-8") as f:
                f.write(content)

if __name__ == "__main__":
    # 获取当前工作目录
    current_dir = Path.cwd()
    print("🚀 开始初始化 Vibe-Synth 项目架构...\n")
    
    # 执行生成
    create_structure(current_dir, PROJECT_STRUCTURE)
    
    print("\n✅ 项目架构初始化完成！")
    print(f"你可以进入目录开始开发: cd {current_dir / 'vibe-synth'}")