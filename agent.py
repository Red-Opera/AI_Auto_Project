import os
import subprocess
import re
import time
from dotenv import load_dotenv
import ollama

load_dotenv()

# 모델 설정
MODEL_NAME = os.getenv("OLLAMA_MODEL_ID", "deepseek-r1:32b")
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
    def __init__(self, project_goal: str = "Create a Sokoban puzzle game"):
        self.max_iterations = 50
        self.project_root = os.getcwd()
        self.project_goal = project_goal
        self.iteration_count = 0

    def analyze_project_state(self, code_override: str = None, error_message: str = None, current_filename_for_context: str = None):
        generated_filename = None 

        if code_override: # 기존 코드 수정 모드
            # 에러가 있는 경우 부분 수정 시도, 없으면 전체 재생성
            if error_message and self._is_suitable_for_partial_fix(error_message):
                return self._request_partial_fix(code_override, error_message, current_filename_for_context)
            else:
                return self._request_full_rewrite(code_override, error_message, current_filename_for_context)
        else: # 새 코드 생성 모드
            return self._request_new_code_generation()

    def _is_suitable_for_partial_fix(self, error_message: str) -> bool:
        """에러가 부분 수정으로 해결 가능한지 판단"""
        # 문법 에러, 간단한 런타임 에러는 부분 수정 가능
        partial_fix_indicators = [
            "SyntaxError",
            "NameError", 
            "AttributeError",
            "TypeError",
            "ValueError",
            "ImportError",
            "ModuleNotFoundError",
            "IndentationError",
            "UnboundLocalError"
        ]
        
        # 타임아웃이나 복잡한 로직 문제는 전체 재작성이 나을 수 있음
        full_rewrite_indicators = [
            "timed out",
            "infinite loop",
            "deadlock",
            "hung",
            "process killed"
        ]
        
        error_lower = error_message.lower()
        
        # 전체 재작성이 필요한 경우
        if any(indicator in error_lower for indicator in full_rewrite_indicators):
            return False
            
        # 부분 수정 가능한 경우
        if any(indicator in error_message for indicator in partial_fix_indicators):
            return True
            
        # 기본적으로 부분 수정 시도
        return True

    def _extract_error_line_info(self, error_message: str) -> str:
        """에러 메시지에서 라인 정보 추출"""
        line_info = ""
        
        # "line X" 패턴 찾기
        line_match = re.search(r'line (\d+)', error_message)
        if line_match:
            line_num = line_match.group(1)
            line_info = f"The error appears to be on or around line {line_num}."
        
        # 파일명과 라인 정보가 함께 있는 경우
        file_line_match = re.search(r'File "([^"]+)", line (\d+)', error_message)
        if file_line_match:
            filename = file_line_match.group(1)
            line_num = file_line_match.group(2)
            line_info = f"The error is in file '{filename}' at line {line_num}."
        
        return line_info

    def _request_partial_fix(self, code: str, error_message: str, filename: str):
        """부분 수정 요청"""
        # 에러 위치 파악
        error_line_info = self._extract_error_line_info(error_message)
        
        prompt = f"""
PROJECT GOAL: {self.project_goal}

Fix ONLY the specific error in the Python code for '{filename}'.
Do NOT rewrite the entire code. Only modify the minimal parts necessary.

ERROR TO FIX:
{error_message}

{error_line_info}

CURRENT CODE:
```python
{code}
```

INSTRUCTIONS:
You must respond with ONLY the corrected code in a single code block.
Fix only what is necessary to resolve the error.
Do not include explanations, markdown formatting outside the code block, or any other text.

Provide the complete corrected code:
"""

        system_prompt = f"""You are an expert Python and Pygame developer.
Fix the specific error in the provided code with minimal changes.

CRITICAL RULES:
- Provide ONLY the complete corrected Python code
- Use a single ```python code block
- No explanations or text outside the code block
- Fix only what's necessary to resolve the reported error
- Maintain the project goal: {self.project_goal}
"""
        
        ai_response = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        # 코드 블록에서 Python 코드 추출
        code_content = self._extract_code_from_response(ai_response)
        
        if code_content and self._is_valid_partial_fix(code, code_content):
            return None, code_content  # 부분 수정 성공
        else:
            # 부분 수정 실패시 전체 재작성으로 폴백
            print("[WARNING] Partial fix failed or invalid, falling back to full rewrite")
            return self._request_full_rewrite(code, error_message, filename)

    def _extract_code_from_response(self, ai_response: str) -> str:
        """AI 응답에서 코드 블록 추출"""
        # 코드 블록 패턴 찾기
        code_block_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'```python(.*?)```',
            r'```(.*?)```'
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, ai_response, re.DOTALL)
            if match:
                code_content = match.group(1).strip()
                if code_content and len(code_content) > 10:
                    return code_content
        
        # 코드 블록이 없으면 응답 전체에서 Python 코드 라인 찾기
        lines = ai_response.split('\n')
        code_lines = []
        in_code_section = False
        
        for line in lines:
            stripped_line = line.strip()
            
            # 명확하게 Python 코드인 라인들
            if (stripped_line.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'pygame.')) or
                re.match(r'^\s*\w+\s*=\s*', stripped_line) and not stripped_line.startswith('#')):
                in_code_section = True
                code_lines.append(line)
            elif in_code_section and (stripped_line == '' or line.startswith('    ') or line.startswith('\t')):
                # 들여쓰기된 라인이나 빈 라인은 코드의 일부로 간주
                code_lines.append(line)
            elif stripped_line.startswith('#') and not stripped_line.startswith('###'):
                # 주석은 포함하되 마크다운 헤더는 제외
                code_lines.append(line)
            elif in_code_section and any(keyword in stripped_line for keyword in ['return', 'break', 'continue', 'pass', 'yield']):
                code_lines.append(line)
            elif stripped_line and not any(marker in stripped_line.lower() for marker in ['instruction:', 'response:', 'here is', 'fix for', '```']):
                # 설명문이 아닌 경우에만 포함
                if in_code_section:
                    code_lines.append(line)
            elif stripped_line and any(marker in stripped_line.lower() for marker in ['instruction:', 'response:', 'here is', 'fix for']):
                # 설명문이 시작되면 코드 섹션 종료
                break
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return None

    def _is_valid_partial_fix(self, original_code: str, fixed_code: str) -> bool:
        """부분 수정이 유효한지 검증"""
        # 기본 검증
        if not fixed_code or len(fixed_code) < 10:
            return False
        
        # 원본 코드와 너무 다르면 부분 수정이 아님
        original_lines = set(original_code.split('\n'))
        fixed_lines = set(fixed_code.split('\n'))
        
        # 공통 라인의 비율 계산
        common_lines = original_lines.intersection(fixed_lines)
        if len(original_lines) > 0:
            similarity_ratio = len(common_lines) / len(original_lines)
            # 50% 이상 유사하면 부분 수정으로 간주
            if similarity_ratio < 0.5:
                print(f"[WARNING] Fixed code too different from original (similarity: {similarity_ratio:.2f})")
                return False
        
        # 문법 검증
        try:
            compile(fixed_code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            print(f"[WARNING] Fixed code has syntax error: {e}")
            return False

    def _apply_partial_changes(self, original_code: str, ai_response: str) -> str:
        """AI의 부분 수정 지시사항을 적용 (레거시 메서드, 현재는 사용되지 않음)"""
        # 이 메서드는 더 이상 사용되지 않지만 호환성을 위해 유지
        return self._extract_code_from_response(ai_response)

    def _parse_change_instructions(self, ai_response: str) -> list:
        """AI 응답에서 변경 지시사항 파싱 (레거시 메서드, 현재는 사용되지 않음)"""
        # 이 메서드는 더 이상 사용되지 않지만 호환성을 위해 유지
        return []

    def _request_full_rewrite(self, code: str, error_message: str, filename: str):
        """전체 코드 재작성 요청"""
        error_context_for_ai = ""
        if error_message:
            if "timed out" in error_message.lower() and \
               "execution of" in error_message.lower() and \
               filename and \
               "sokoban" in self.project_goal.lower():
                error_context_for_ai = f"""
The Python script '{filename}' started execution but did not finish within the {self.get_execution_timeout()} second time limit.
This timeout likely means the game initialized and entered its main loop.

Fix any issues and ensure the code:
1. Initializes Pygame correctly
2. Has a proper game loop with event handling
3. Handles pygame.QUIT events to allow graceful exit
4. Runs without immediate errors or hangs

Error: {error_message}
"""
            else:
                error_context_for_ai = f"Fix this error:\n{error_message}"

        prompt = f"""
PROJECT GOAL: {self.project_goal}

Fix the Python code for '{filename}' to resolve the error and achieve the project goal.
{error_context_for_ai}

Current code:
```python
{code}
```

Provide ONLY the complete corrected Python code in a single code block. No explanations.
"""
        system_prompt = f"""You are an expert Python and Pygame developer. 
Fix the provided code to achieve: {self.project_goal}

REQUIREMENTS:
- Provide complete, working Python code
- Use a single ```python code block
- No explanations or text outside the code block
- Fix all errors and ensure the code runs properly
"""
        
        ai_response_content = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        # 코드 블록에서 Python 코드 추출
        code_content = self._extract_code_from_response(ai_response_content)
        
        return None, code_content

    def _request_new_code_generation(self):
        """새 코드 생성 요청"""
        files = self.get_project_files()
        structure = self.get_directory_structure()
        prompt = f"""
PROJECT GOAL: {self.project_goal}

Current project state:
Files: {files}
Directory structure: {structure}

Create complete Python code from scratch to achieve the project goal.

At the VERY BEGINNING of your response, include: # filename: chosen_filename.py
Then provide ONLY Python code in a code block, no explanations.
"""
        system_prompt = f"""You are an expert Python developer. 
Create code to achieve: {self.project_goal}

REQUIREMENTS:
- Start with: # filename: chosen_filename.py
- Provide complete Python code in ```python code blocks
- No explanations outside the code block
- Ensure the code achieves the project goal"""
        
        ai_response_content = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        generated_filename = None
        code_content = None

        # 파일명 파싱
        filename_match = re.search(r'^#\s*filename:\s*([\w\.-]+\.py)\s*$', ai_response_content, re.IGNORECASE | re.MULTILINE)
        if filename_match:
            generated_filename = filename_match.group(1).strip()
            # 파일명 라인 이후의 내용만 코드 추출 대상으로 함
            ai_response_content = ai_response_content[filename_match.end():].strip()
        else:
            print("[WARNING] AI did not provide a filename in the expected format for new code.")
        
        # 코드 블록에서 Python 코드 추출
        code_content = self._extract_code_from_response(ai_response_content)
        
        if not code_content or len(code_content) < 10:
            print("[WARNING] Could not extract valid Python code from AI response.")
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
    # 프로젝트 목표를 소코반 게임으로 변경
    project_goal = "Create a functional Sokoban puzzle game using pygame. The game should include a player character that can move around, boxes that can be pushed, target locations where boxes need to be placed, and walls. The game must use only basic geometric shapes (rectangles, circles) drawn with pygame.draw functions for all visual elements. Include basic collision detection, win condition checking when all boxes are on targets, and keyboard controls (arrow keys or WASD). No external image files or other asset files should be loaded or used."
    agent = AutonomousAgent(project_goal=project_goal)
    # 초기 코드나 파일명 없이 시작하여 AI가 생성하도록 함
    agent.run_autonomous_loop()
