import os
import subprocess
import re
import time
from dotenv import load_dotenv
import ollama

load_dotenv()

# 모델 설정
MODEL_NAME = os.getenv("OLLAMA_MODEL_ID", "granite-code:20b")
TRANSLATION_MODEL = os.getenv("OLLAMA_TRANSLATION_MODEL", "llama2")

# 명령어 실행 함수
def run_command(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    output, error = process.communicate()
    return output, error

# 텍스트를 영어로 번역하는 함수
def translate_to_english(text: str) -> str:
    response = ollama.chat(
        model=TRANSLATION_MODEL,
        messages=[
            {"role": "system", "content": "You are a translation assistant. Translate the user-provided Korean text into fluent English."},
            {"role": "user",   "content": text}
        ]
    )
    return response.get("message", {}).get("content", "")

# AI 호출 함수
def call_ai(system_prompt: str, user_prompt: str, translate: bool = False) -> str:
    if translate:
        user_prompt = translate_to_english(user_prompt)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    return response.get("message", {}).get("content", "")

# 테스트 실행 함수
def run_tests():
    try:
        result = subprocess.run(
            ["python", "-m", "pytest"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return (result.stdout or "") + (result.stderr or "")
    except Exception:
        return "테스트 실행 실패"

class AutonomousAgent:
    def __init__(self, project_goal: str = "Create a simple Flappy Bird game"):
        self.max_iterations = 50
        self.project_root = os.getcwd()
        self.project_goal = project_goal
        self.iteration_count = 0

    def analyze_project_state(self, code_override: str = None, error_message: str = None, current_filename_for_context: str = None):
        generated_filename = None 

        if code_override: # 기존 코드 수정 모드
            error_context_for_ai = ""
            if error_message:
                # Pygame 프로젝트에서 실행 타임아웃이 발생한 경우 특별한 컨텍스트 제공
                if "timed out" in error_message.lower() and \
                   "execution of" in error_message.lower() and \
                   current_filename_for_context and \
                   "pygame" in self.project_goal.lower():
                    error_context_for_ai = f"""
The Python script '{current_filename_for_context}' started execution but did not finish within the {self.get_execution_timeout()} second time limit.
This is a common situation for Pygame applications, as they typically enter a game loop that runs until the user quits.
This timeout likely means the game initialized and entered its main loop, but the automated test had to stop it.

Please review the code, especially the Pygame initialization, event handling, and the main game loop.
Ensure that:
1. Pygame initializes correctly without any immediate errors (e.g., all necessary modules are imported and initialized, resources are handled if any were intended even with basic shapes).
2. The main game loop correctly handles events, especially the `pygame.QUIT` event to allow the game to close gracefully.
3. There are no unintentional infinite loops or deadlocks during the initialization phase or in the very early stages of the game loop that would prevent it from running smoothly for a few seconds.
4. The game is structured to be runnable. For testing purposes, it should at least reach a state where it's clear the main loop has started.

The specific error reported by the test environment was:
{error_message}

Your task is to refine the code to ensure it's a robust, runnable Pygame application that adheres to the project goal.
Focus on making sure the game can start, run its loop, and handle basic exit conditions.
"""
                else:
                    error_context_for_ai = f"The code has the following error that needs to be fixed:\n{error_message}"

            prompt = f"""
PROJECT GOAL: {self.project_goal}

You are improving the following Python code for the file '{current_filename_for_context if current_filename_for_context else "the project file"}'.
{error_context_for_ai}

Current code:
```python
{code_override}
```
You MUST fix any reported errors and improve the code to achieve the project goal.
Provide ONLY the corrected and complete Python code in a single code block. No explanations.
"""
            system_prompt = f"""You are an expert Python and Pygame developer. 
Your task is to improve Python code to achieve this specific goal: {self.project_goal}

REQUIREMENTS:
- Always provide complete, working Python code.
- Code must be enclosed in a single ```python code block.
- No explanations, comments, or text outside the code block.
- Focus on creating functional, clean implementations that are robust.
- Ensure the code achieves the specified project goal and can run without immediate errors or hangs.
- If previous code timed out during execution, ensure the new code initializes correctly and the game loop is well-structured.
"""
        else: # 새 코드 생성 모드
            files = self.get_project_files()
            structure = self.get_directory_structure()
            prompt = f"""
PROJECT GOAL: {self.project_goal}

Current project state:
Files: {files}
Directory structure: {structure}

You must create complete Python code from scratch to achieve the project goal.
This is the initial code creation - make it comprehensive and functional.

At the VERY BEGINNING of your response, before the Python code block, include a line specifying the target filename, like this:
# filename: chosen_filename.py

Then, provide ONLY Python code in a code block, no explanations.
The code must be complete and ready to run.
"""
            system_prompt = f"""You are an expert Python developer. 
Your task is to create code to achieve this specific goal: {self.project_goal}

REQUIREMENTS:
- At the VERY BEGINNING of your response, include a line like: # filename: chosen_filename.py
- Then, provide complete, working Python code enclosed in ```python code blocks
- No other explanations, comments, or text outside the code block
- Focus on creating functional, clean implementations
- Ensure the code achieves the specified project goal"""
        
        ai_response_content = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        code_content = None

        if not code_override: # 새 코드 생성 시에만 파일명 파싱 시도
            filename_match = re.search(r'^#\s*filename:\s*([\w\.-]+\.py)\s*$', ai_response_content, re.IGNORECASE | re.MULTILINE)
            if filename_match:
                generated_filename = filename_match.group(1).strip()
                # 코드 파싱을 위해 파일명 줄 제거
                ai_response_content = ai_response_content[filename_match.end():].strip()
            else:
                print("[WARNING] AI did not provide a filename in the expected format for new code.")
        
        # 코드 블록에서 Python 코드 추출
        match = re.search(r'```(?:python)?\n(.*?)\n```', ai_response_content, re.DOTALL)
        if match:
            code_content = match.group(1).strip()
        else:
            # 코드 블록이 없으면 전체 응답에서 코드 같은 부분 찾기 (Fallback)
            lines = ai_response_content.split('\n')
            code_lines = []
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith(('import ', 'from ', 'def ', 'class ')) or \
                   (re.match(r'^\s*(if|for|while|try|except|finally|with|return|yield|pass|break|continue)', stripped_line)) or \
                   (re.match(r'^\s*\w+\s*=\s*', stripped_line) and not stripped_line.startswith('#')):
                    code_lines.append(line)
            
            if code_lines:
                code_content = '\n'.join(code_lines).strip()
            
            if not code_content or len(code_content) < 10:
                print("[WARNING] Could not find a valid Python code block or substantial code lines in AI response.")
                code_content = None

        return generated_filename, code_content

    def apply_code(self, code: str, filename: str):
        if not code:
            print("[ERROR] No code to apply")
            return False
        if not filename:
            print("[ERROR] No filename provided to apply code")
            return False
            
        target_file = filename # 파라미터로 받은 파일명 사용
        try:
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"[INFO] Applied code to {target_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write file {target_file}: {e}")
            return False

    def test_code(self, filename: str):
        """코드의 문법 및 실행 가능성을 테스트합니다."""
        try:
            # 1. 문법 검사 (컴파일 시도)
            with open(filename, 'r', encoding='utf-8') as f:
                pass 
            
            compile_process = subprocess.run(
                ["python", "-m", "py_compile", filename],
                capture_output=True,
                text=True, 
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )

            if compile_process.returncode != 0:
                error_message = compile_process.stderr or compile_process.stdout
                detailed_error = f"Syntax error during compilation: {error_message}"
                return False, detailed_error

            # 2. 실제 실행 테스트
            exec_process = subprocess.run(
                ["python", filename],
                capture_output=True,
                text=True, 
                timeout=5, # 실행 타임아웃은 짧게 유지하여 빠른 피드백
                encoding='utf-8',
                errors='replace'
            )

            # 실행 성공 (returncode 0)
            if exec_process.returncode == 0:
                # stderr에 Traceback이 있는지 확인 (간혹 returncode 0 이어도 에러 출력하는 경우)
                if "Traceback (most recent call last):" in exec_process.stderr:
                    return False, f"Runtime error (stderr despite exit code 0):\n{exec_process.stderr}"
                
                # stdout에 Pygame 환영 메시지가 있는지 확인
                pygame_hello_message = "Hello from the pygame community"
                if pygame_hello_message in exec_process.stdout:
                    return True, "Syntax OK. Pygame initialized successfully and script exited gracefully within timeout."
                return True, "Syntax OK and Execution OK (short test, script exited gracefully)."
            # 실행 실패 (returncode != 0)
            else:
                error_output = exec_process.stderr if exec_process.stderr else exec_process.stdout
                return False, f"Runtime error during execution (exit code {exec_process.returncode}):\n{error_output}"

        except subprocess.TimeoutExpired as e:
            command_list = e.cmd
            
            # 컴파일 타임아웃
            if command_list and len(command_list) > 2 and command_list[1] == "-m" and command_list[2] == "py_compile":
                return False, f"Syntax check error: Compilation (py_compile) timed out after {e.timeout} seconds for '{filename}'."
            
            # 실행 타임아웃
            elif command_list and len(command_list) > 1 and command_list[0] == "python" and command_list[1] == filename:
                stderr_content = e.stderr if e.stderr else ""
                stdout_content = e.stdout if e.stdout else ""
                
                error_detail = ""
                pygame_initialized_message = ""

                # 타임아웃 전 stdout에서 Pygame 환영 메시지 확인
                if "Hello from the pygame community" in stdout_content:
                    pygame_initialized_message = "Note: Pygame appears to have initialized successfully (based on 'Hello from the pygame community' message in stdout) before the timeout."

                if stderr_content and "Traceback (most recent call last):" in stderr_content:
                    error_detail = f"A Python traceback was found in stderr before timeout:\n{stderr_content}"
                elif stdout_content and "Traceback (most recent call last):" in stdout_content: # 드문 경우
                    error_detail = f"A Python traceback was found in stdout before timeout:\n{stdout_content}"
                elif stderr_content: # Traceback은 없지만 stderr에 다른 내용이 있는 경우
                    error_detail = f"Stderr output before timeout:\n{stderr_content}"
                # stdout은 Pygame 메시지 외 다른 내용이 있을 수 있으나, 에러 판단에는 stderr 우선
                elif stdout_content and not pygame_initialized_message: # Pygame 메시지 외 다른 stdout 내용
                     error_detail = f"Stdout output before timeout:\n{stdout_content}"


                timeout_reason_message = f"Runtime error: Code execution of '{filename}' timed out after {e.timeout} seconds."
                
                final_message_parts = [timeout_reason_message]
                if pygame_initialized_message:
                    final_message_parts.append(pygame_initialized_message)
                if error_detail:
                    final_message_parts.append(error_detail)
                    final_message_parts.append("The timeout might be a consequence of the issue detailed above, or due to the game loop running as expected.")
                else: # 특정 에러 없이 타임아웃
                    final_message_parts.append("No specific Python traceback was captured before the timeout. The script might have an infinite loop, be waiting for input/resources, or the game loop is running as expected and exceeded the test duration.")
                
                return False, "\n".join(final_message_parts)
            else: # 알 수 없는 명령어의 타임아웃
                return False, f"An unspecified command timed out after {e.timeout} seconds: {' '.join(command_list if command_list else ['Unknown command'])}"
        except FileNotFoundError:
            return False, f"Error: The file '{filename}' was not found for testing."
        except Exception as e: # 그 외 예외
            return False, f"An unexpected error occurred during testing of '{filename}': {str(e)}"

    def get_directory_structure(self):
        lines = []
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d != '.git']
            level = root.replace('.', '').count(os.sep)
            indent = ' ' * 2 * level
            lines.append(f"{indent}{os.path.basename(root)}/")
            for f in files:
                if not f.startswith('.'):
                    lines.append(f"{indent}  {f}")
        return '\n'.join(lines)

    def get_project_files(self):
        files = []
        for root, dirs, filenames in os.walk('.'):
            dirs[:] = [d for d in dirs if d != '.git']
            for fn in filenames:
                if not fn.startswith('.') and fn.endswith('.py'):
                    files.append(os.path.join(root, fn))
        return '\n'.join(files)

    def get_execution_timeout(self):
        # test_code 함수에 설정된 실행 타임아웃 값을 반환 (일관성 유지)
        # 이 값을 test_code와 동기화해야 합니다.
        return 5 # 현재 test_code의 실행 타임아웃이 5초로 하드코딩되어 있음

    def run_autonomous_loop(self, initial_code: str = None, initial_filename: str = None):
        current_code = initial_code
        current_filename = initial_filename
        last_error = None
        
        for i in range(self.max_iterations):
            self.iteration_count = i + 1
            print(f"[INFO] Iteration {self.iteration_count}/{self.max_iterations}")
            
            # AI 호출하여 코드 및 파일명 제안 받기
            suggested_filename_from_ai, new_code = self.analyze_project_state(
                code_override=current_code, 
                error_message=last_error,
                current_filename_for_context=current_filename # 현재 파일명 컨텍스트 전달
            )

            if not new_code:
                print("[ERROR] AI failed to generate code.")
                if not current_code : # 초기 생성 실패 시 더 이상 진행 불가
                    print("[ERROR] Critical failure: No initial code generated.")
                    break
                # 기존 코드가 있다면, 에러 상태로 다음 반복 시도 가능 (또는 다른 전략)
                print("[INFO] Retrying with existing code due to AI generation failure.")
                time.sleep(1)
                continue


            target_filename_for_this_iteration = current_filename

            if suggested_filename_from_ai: 
                # AI가 새 파일명을 제안한 경우 (주로 초기 생성 시)
                target_filename_for_this_iteration = suggested_filename_from_ai
            elif not current_filename: 
                # AI가 파일명 제안 안했고, 기존 파일명도 없는 경우 (초기 생성 시 AI가 파일명 누락)
                print("[WARNING] AI did not suggest a filename for initial creation. Using default 'ai_generated_code.py'.")
                target_filename_for_this_iteration = "ai_generated_code.py"
            # else: AI가 파일명 제안 안했고, 기존 파일명 있으면 current_filename 사용 (위에서 이미 설정됨)
            
            if not target_filename_for_this_iteration:
                print("[ERROR] Target filename could not be determined. Aborting iteration.")
                break

            # 코드 적용
            if not self.apply_code(new_code, target_filename_for_this_iteration):
                print(f"[ERROR] Failed to apply code to {target_filename_for_this_iteration}. Aborting loop.")
                # 파일 쓰기 실패는 심각한 문제로 간주하고 중단
                break 
            
            # 코드 테스트
            is_valid, test_result = self.test_code(target_filename_for_this_iteration)
            print(f"[TEST] File '{target_filename_for_this_iteration}': {test_result}")
            
            if is_valid:
                print(f"[SUCCESS] Code in '{target_filename_for_this_iteration}' is syntactically correct!")
                current_code = new_code
                current_filename = target_filename_for_this_iteration # 현재 파일명 및 코드 업데이트
                last_error = None
                
                print(f"[INFO] Successfully generated/updated working code in '{current_filename}'!")
                break # 성공 시 루프 종료
            else:
                print(f"[ERROR] Code in '{target_filename_for_this_iteration}' has issues: {test_result}")
                last_error = test_result
                # 실패한 코드와 파일명을 다음 반복을 위해 유지
                current_code = new_code 
                current_filename = target_filename_for_this_iteration
            
            time.sleep(1)
        
        print(f"[INFO] Autonomous loop completed after {self.iteration_count} iterations.")
        if current_filename and current_code and last_error is None:
             print(f"Final working code is in: {current_filename}")
        elif current_filename:
             print(f"Last attempted code (may have errors) is in: {current_filename}")
        else:
             print("No code was successfully processed or generated.")

if __name__ == '__main__':
    # 프로젝트 목표를 명확히 설정
    project_goal = "Create a functional Flappy Bird game using pygame with a bird that can jump and basic physics. The game must use only basic geometric shapes (e.g., rectangles, circles) drawn with pygame.draw functions for all visual elements including the bird, pipes, and background. No external image files or other asset files should be loaded or used."
    agent = AutonomousAgent(project_goal=project_goal)
    # 초기 코드나 파일명 없이 시작하여 AI가 생성하도록 함
    agent.run_autonomous_loop()
