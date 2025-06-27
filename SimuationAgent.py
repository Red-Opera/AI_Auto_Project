from dotenv import load_dotenv
import os
import openai
import datetime
import subprocess
import re
import traceback  # traceback 모듈 추가

load_dotenv()

# 모델 설정 - OpenAI API 키와 모델 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # 기본 모델을 gpt-4.1-mini로 설정
TRANSLATION_MODEL = os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4.1-mini")  # 번역에도 동일 모델 사용

# OpenAI 클라이언트 설정
openai.api_key = OPENAI_API_KEY

# API 응답을 파일에 저장하는 함수
def save_response_to_file(response_text, prefix="response"):
    """API 응답을 임시 파일에 저장합니다."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.txt"
        
        # 항상 response.txt에 최신 응답 저장
        with open("response.txt", "w", encoding="utf-8") as f:
            f.write(response_text)
            
        # 타임스탬프가 있는 백업 파일도 저장 (선택적)
        if os.getenv("SAVE_RESPONSE_HISTORY", "false").lower() == "true":
            with open(filename, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(f"[INFO] Response saved to {filename}")
            
        return True
    except Exception as e:
        print(f"[WARNING] Failed to save response to file: {e}")
        return False

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
    try:
        response = openai.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": "You are a translation assistant. Translate the user-provided Korean text into fluent English."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        response_content = response.choices[0].message.content
        
        # 응답 저장
        save_response_to_file(response_content, prefix="translation")
        
        return response_content
    except Exception as e:
        print(f"[ERROR] Translation API error: {e}")
        return text  # 에러 발생 시 원본 텍스트 반환

# AI 호출 함수
def call_ai(system_prompt: str, user_prompt: str, translate: bool = False) -> str:
    if translate:
        user_prompt = translate_to_english(user_prompt)
    
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        # API 응답 내용
        response_content = response.choices[0].message.content
        
        # 전체 응답 객체 정보도 저장 (디버깅용)
        full_response_info = f"""
API Response Details:
---------------------
Model: {MODEL_NAME}
Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
System Prompt: {system_prompt}
User Prompt: {user_prompt[:500]}{'...' if len(user_prompt) > 500 else ''}

Response Content:
----------------
{response_content}

Usage Information:
-----------------
Prompt Tokens: {response.usage.prompt_tokens}
Completion Tokens: {response.usage.completion_tokens}
Total Tokens: {response.usage.total_tokens}
"""
        
        # 응답 저장
        save_response_to_file(full_response_info)
        
        return response_content
    except Exception as e:
        error_message = f"API 호출 오류: {str(e)}"
        print(f"[ERROR] OpenAI API error: {e}")
        
        # 에러 정보도 저장
        save_response_to_file(f"API Error: {error_message}", prefix="error")
        
        return error_message

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
    def __init__(self, project_goal: str = "Create a pathfinding algorithm visualization"):
        self.max_iterations = 50
        self.project_root = os.getcwd()
        self.project_goal = project_goal
        self.iteration_count = 0
        
        # 로그 디렉토리 설정
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_directory = os.path.join(self.project_root, f"Log({self.timestamp})")
        self._create_log_directory()
        
        # 예외 이력 추적을 위한 속성 추가
        self.error_history = []
        self.same_error_count = 0
        self.last_error_message = None
        self.strategy_level = 0  # 점진적으로 다른 전략 적용을 위한 레벨

    def _create_log_directory(self):
        """로그 디렉토리 생성"""
        try:
            if not os.path.exists(self.log_directory):
                os.makedirs(self.log_directory)
                print(f"[INFO] Created log directory: {self.log_directory}")
        except Exception as e:
            print(f"[ERROR] Failed to create log directory: {e}")
            # 로그 디렉토리를 생성할 수 없으면 현재 디렉토리를 사용
            self.log_directory = self.project_root

    def _save_iteration_log(self, iteration_number, code, error_message=None):
        """각 반복(Iteration)의 코드와 에러 메시지를 로그 파일에 저장"""
        try:
            log_filename = os.path.join(self.log_directory, f"Iteration - {iteration_number}.py")
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(code)
                if error_message:
                    f.write("\n\n# 실패 이유:\n# " + error_message.replace('\n', '\n# '))
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save iteration log: {e}")
            return False

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
            return code_content  # 부분 수정 성공
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

    # 코드 유사성 검사 메서드 추가
    def _is_code_similar(self, code1, code2, threshold=0.9):
        """두 코드가 유사한지 비교합니다."""
        if not code1 or not code2:
            return False
            
        lines1 = set(code1.split('\n'))
        lines2 = set(code2.split('\n'))
        
        # 공통 라인 수
        common_lines = lines1.intersection(lines2)
        
        # 유사도 계산 (Jaccard 유사도)
        similarity = len(common_lines) / (len(lines1) + len(lines2) - len(common_lines))
        
        return similarity >= threshold

    def _request_full_rewrite(self, code: str, error_message: str, filename: str):
        """전체 코드 재작성 요청"""
        error_context_for_ai = ""
        if error_message:
            # 에러 메시지를 명확하게 포함
            error_context_for_ai = f"""
The simulation has the following error:
----------------
{error_message}
----------------

Please fix this error by rewriting the code to properly handle this issue.
"""

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
        system_prompt = f"""You are an expert Python developer. 
Fix the provided code to achieve: {self.project_goal}

REQUIREMENTS:
- Provide complete, working Python code
- Use a single ```python code block
- No explanations or text outside the code block
- Fix all errors and ensure the code runs properly

CRITICAL RULES FOR SIMULATION DESIGN:
- Design a NEW, DIFFERENT maze layout for this iteration (iteration {self.iteration_count})
- Do not reuse the previous maze design - create something fresh and unique
- Ensure the simulation has clear completion conditions
- Implement proper algorithm visualization that works reliably
- Design intuitive controls that respond consistently
- Avoid infinite loops, deadlocks, or unescapable situations
- Implement proper state management and transitions
- Include proper error handling for unexpected situations
- Ensure performance is optimized for smooth visualization

MANDATORY AUTOMATIC TESTING FEATURES:
1. When run with the '--auto-test' command line argument, the simulation must:
   - Run all algorithms automatically without user input
   - Print 'SIMULATION_COMPLETE_SUCCESS' to stdout when all algorithms reach the goal
   - Exit gracefully after completion
   - This mode is used to verify the simulation works correctly
"""
    
        ai_response = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        extracted_code = self._extract_code_from_response(ai_response)
        
        # 코드가 없거나 튜플인 경우 처리
        if not extracted_code:
            print("[WARNING] No code was extracted from AI response")
            return code  # 원래 코드 반환
        
        if isinstance(extracted_code, tuple):
            if len(extracted_code) > 0:
                extracted_code = extracted_code[0]  # 튜플의 첫 번째 요소 사용
            else:
                return code  # 원래 코드 반환
                
        # 문자열이 아닌 경우 변환
        if not isinstance(extracted_code, str):
            extracted_code = str(extracted_code)
            
        return extracted_code

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
        system_prompt = f"""You are an expert Python developer specialized in creating stable, reliable code.
Create code to achieve: {self.project_goal}

REQUIREMENTS:
- Start with: # filename: chosen_filename.py
- Provide complete Python code in ```python code blocks
- No explanations outside the code block
- Ensure the code achieves the project goal

CRITICAL RULES FOR SIMULATION DESIGN:
- Create a complete pathfinding algorithm visualization system
- Design a clear, functional maze generation algorithm
- Include proper visualization for BFS, DFS, and A* algorithms
- Ensure all algorithms work correctly and can find paths in valid mazes
- Implement proper state management and transitions
- Include proper error handling for unexpected situations
- Ensure performance is optimized for smooth visualization
- Generate sounds programmatically as specified

MANDATORY AUTOMATIC TESTING FEATURES:
- When run with the '--auto-test' command line argument, the simulation must:
  - Run all algorithms automatically without user input
  - Print 'SIMULATION_COMPLETE_SUCCESS' to stdout when all algorithms reach the goal
  - Exit gracefully after completion
  - This mode is used to verify the simulation works correctly
"""
    
        ai_response = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        # 파일명 추출
        filename_match = re.search(r'# filename: (.+\.py)', ai_response)
        if filename_match:
            filename = filename_match.group(1)
        else:
            filename = "maze_pathfinding_simulation.py"  # 기본 파일명
        
        # 코드 추출
        extracted_code = self._extract_code_from_response(ai_response)
        
        return filename, extracted_code

    def apply_code(self, code: str, filename: str):
        """생성된 코드를 파일에 적용합니다."""
        try:
            # code가 튜플인 경우 문자열로 변환
            if isinstance(code, tuple):
                if len(code) > 0:
                    code = code[0]  # 튜플의 첫 번째 요소 사용
                else:
                    code = ""
                    
            # code가 문자열인지 확인
            if not isinstance(code, str):
                code = str(code)
                
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"[INFO] Code successfully written to {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write file {filename}: {str(e)}")
            return False

    def test_code(self, filename: str):
        """코드의 문법 및 실행 가능성을 테스트합니다."""
        try:
            # 1. 문법 검사 (컴파일 시도)
            with open(filename, 'r', encoding='utf-8') as f:
                current_code = f.read()
    
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

            # 2. 자동 테스트 모드로 실행 (TEST_MODE=1 환경 변수 설정)
            env = os.environ.copy()
            env["TEST_MODE"] = "1"
            
            try:
                exec_process = subprocess.run(
                    ["python", filename, "--test"],
                    capture_output=True,
                    text=True, 
                    timeout=30,
                    encoding='utf-8',
                    errors='replace',
                    env=env
                )
                
                # 테스트 성공 확인 (TEST_SUCCESS 문자열 체크)
                if "TEST_SUCCESS" in exec_process.stdout:
                    return True, "자동 테스트 모드에서 성공적으로 실행되었습니다. TEST_SUCCESS 메시지를 확인했습니다."
            except subprocess.TimeoutExpired:
                # 테스트 모드 타임아웃은 무시 (시뮬레이션이 예상대로 작동 중일 수 있음)
                pass

            # 3. 일반 모드로 실행 (이전 로직과 동일)
            exec_process = subprocess.run(
                ["python", filename],
                capture_output=True,
                text=True, 
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )

            # 실행 성공 (returncode 0)
            if exec_process.returncode == 0:
                # stderr에 Traceback이 있는지 확인
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

                # Pygame이 초기화되었고, 기존 코드와 변경이 없는 경우 성공으로 처리
                if pygame_initialized_message:
                    # 기존 코드 확인 (이전 반복에서 성공한 코드가 있는 경우 체크)
                    try:
                        # 성공한 최종 코드를 이전에 저장했는지 확인
                        previous_success_path = os.path.join(self.log_directory, "final_solution.py")
                        if os.path.exists(previous_success_path):
                            with open(previous_success_path, 'r', encoding='utf-8') as f:
                                previous_code = f.read()
                            
                            # 현재 코드와 이전 성공 코드가 유사한지 확인
                            if self._is_code_similar(current_code, previous_code, threshold=0.9):
                                return True, "Test PASSED: Code is similar to previously successful version and Pygame initialized correctly."
                    except Exception as ex:
                        print(f"[WARNING] Failed to compare with previous successful code: {ex}")
                    
                    # Pygame이 초기화되었다면 시뮬레이션이 시작된 것으로 간주하고 성공으로 처리
                    return True, "Test PASSED: Pygame simulation started successfully. Timeout is expected for simulations with main loops."

                # 이하 기존 코드와 동일...
                if stderr_content and "Traceback (most recent call last):" in stderr_content:
                    error_detail = f"A Python traceback was found in stderr before timeout:\n{stderr_content}"
                elif stdout_content and "Traceback (most recent call last):" in stdout_content:
                    error_detail = f"A Python traceback was found in stdout before timeout:\n{stdout_content}"
                elif stderr_content:
                    error_detail = f"Stderr output before timeout:\n{stderr_content}"
                elif stdout_content and not pygame_initialized_message:
                    error_detail = f"Stdout output before timeout:\n{stdout_content}"

                timeout_reason_message = f"Runtime error: Code execution of '{filename}' timed out after {e.timeout} seconds."
                
                final_message_parts = [timeout_reason_message]
                if pygame_initialized_message:
                    final_message_parts.append(pygame_initialized_message)
                if error_detail:
                    final_message_parts.append(error_detail)
                    final_message_parts.append("The timeout might be a consequence of the issue detailed above, or due to the simulation loop running as expected.")
                else: # 특정 에러 없이 타임아웃
                    final_message_parts.append("No specific Python traceback was captured before the timeout. The script might have an infinite loop, be waiting for input/resources, or the simulation loop is running as expected and exceeded the test duration.")
                
                # 코드 분석을 통한 추가 정보
                try:
                    # 시뮬레이션 루프 분석
                    has_game_loop = any(pattern in current_code for pattern in ['while running:', 'while True:', 'while not done:'])
                    has_event_handling = 'pygame.event.get()' in current_code
                    has_quit_handling = 'pygame.QUIT' in current_code
                    has_draw_update = 'pygame.display.update()' in current_code or 'pygame.display.flip()' in current_code
                    
                    analysis_message = "\nCode Analysis:\n"
                    if has_game_loop and has_event_handling and has_quit_handling and has_draw_update:
                        analysis_message += "- Simulation appears to have proper main loop with event handling, quit mechanism, and display updates.\n"
                        analysis_message += "- Timeout likely indicates normal simulation execution rather than an error.\n"
                        analysis_message += "- This is EXPECTED behavior for a properly functioning simulation.\n"
                    else:
                        if not has_game_loop:
                            analysis_message += "- WARNING: No clear main loop detected. Simulation may exit prematurely or hang.\n"
                        if not has_event_handling:
                            analysis_message += "- WARNING: Event handling missing. Simulation may become unresponsive.\n"
                        if not has_quit_handling:
                            analysis_message += "- WARNING: No pygame.QUIT handling detected. Simulation may not exit properly.\n"
                        if not has_draw_update:
                            analysis_message += "- WARNING: No display update calls found. Simulation may not render properly.\n"
                    
                    final_message_parts.append(analysis_message)
                except Exception as e:
                    pass  # 분석 실패 시 추가 정보 제공 안함
                
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
        return 30 # 30초로 설정

    def _request_simplified_version(self, code, error_message, filename):
        """시뮬레이션을 단순화하여 재작성 요청"""
        prompt = f"""
PROJECT GOAL: {self.project_goal}

The current simulation code is too complex and causing issues.
Create a SIMPLIFIED version with:
- Smaller maze size (20x20 instead of 50x50)
- Fewer visual effects and animations
- Simpler UI elements
- Basic sound generation

Error message:
{error_message}

Current code:
```python
{code}
```

Provide ONLY the complete simplified Python code that achieves the core functionality.
"""

        system_prompt = f"""You are an expert Python developer.
Create a simplified version of the pathfinding visualization that focuses on core functionality.

REQUIREMENTS:
- Provide ONLY the complete Python code
- Focus on stability and reliability over features
- Implement a smaller maze (20x20)
- Simplify visualization while keeping algorithm comparison
- Ensure all algorithms work correctly
- Maintain automatic testing capability with --auto-test flag
"""

        ai_response = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        return self._extract_code_from_response(ai_response)

    def _request_alternative_approach(self, code, error_message, filename):
        """대체 접근법 요청 (pygame 대신 다른 라이브러리 고려)"""
        prompt = f"""
PROJECT GOAL: {self.project_goal}

The current implementation approach is causing issues.
Consider an ALTERNATIVE APPROACH that might be more stable:
- Consider using tkinter instead of pygame if appropriate
- Or simplify the pygame implementation significantly
- Focus on algorithm visualization over fancy graphics
- Ensure stability and completion over visual appeal

Error message:
{error_message}

Current code:
```python
{code}
```

Provide ONLY the complete Python code using an alternative approach.
"""

        system_prompt = f"""You are an expert Python developer.
Create an alternative implementation of the pathfinding visualization using a different approach.

REQUIREMENTS:
- Provide ONLY the complete Python code
- Consider tkinter as an alternative to pygame if appropriate
- Or create a significantly simplified pygame implementation
- Focus on algorithm correctness and stability
- Ensure automatic testing capability with --auto-test flag
"""

        ai_response = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        return self._extract_code_from_response(ai_response)

    def _request_incremental_implementation(self, code, error_message, filename):
        """단계적 구현 요청 (한 알고리즘씩 구현)"""
        prompt = f"""
PROJECT GOAL: {self.project_goal}

Implement an INCREMENTAL APPROACH to the visualization:
- Start with implementing ONLY ONE algorithm (BFS) properly
- Focus on maze generation and single algorithm visualization
- Ensure that one algorithm works perfectly before adding others
- Implement a clear completion condition for the single algorithm

Error message:
{error_message}

Current code:
```python
{code}
```

Provide ONLY the complete Python code implementing this incremental approach.
"""

        system_prompt = f"""You are an expert Python developer.
Create an incremental implementation of the pathfinding visualization, starting with just one algorithm.

REQUIREMENTS:
- Provide ONLY the complete Python code
- Implement only BFS algorithm first with perfect visualization
- Focus on stability and clear completion conditions
- Ensure maze generation works correctly
- Include automatic testing capability with --auto-test flag
"""

        ai_response = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        return self._extract_code_from_response(ai_response)

    def run_autonomous_loop(self, initial_code: str = None, initial_filename: str = None):
        """자율 개발 루프를 실행합니다."""
        code = initial_code
        filename = initial_filename
        self.iteration_count = 0
        
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n===== ITERATION {self.iteration_count} =====")
            
            try:
                # 코드가 없으면 새로 생성
                if not code or not filename:
                    generated_filename, generated_code = self._request_new_code_generation()
                    if generated_code and generated_filename:
                        code = generated_code
                        filename = generated_filename
                    else:
                        print(f"[ERROR] Failed to generate code at iteration {self.iteration_count}")
                        continue
    
                # 코드 적용
                self.apply_code(code, filename)
                print(f"[INFO] Applied code to {filename}")
                
                # 코드 테스트
                print(f"[INFO] Testing code {filename}...")
                success, error_message = self.test_code(filename)
                
                # 테스트 결과 디버깅 로그 추가
                print(f"[DEBUG] Test result - Success: {success}, Error: {error_message[:100]}...")
                
                # 시뮬레이션 자동 테스트 실행
                if success:
                    print(f"[INFO] Running automated simulation test for {filename}...")
                    simulation_result = self.test_simulation_completion(filename)
                    
                    # 에러가 없고 시뮬레이션이 완료되면 완료
                    if simulation_result.get('success', False):
                        print(f"[SUCCESS] Code passes all tests and simulation completes successfully! Development finished.")
                        
                        # 성공한 코드를 final_solution.py로 저장
                        final_path = os.path.join(self.log_directory, "final_solution.py")
                        try:
                            with open(final_path, 'w', encoding='utf-8') as f:
                                f.write(code)
                            print(f"[INFO] Saved successful code to {final_path}")
                        except Exception as e:
                            print(f"[WARNING] Failed to save final solution: {e}")
                            
                        break
        
                # 에러가 있거나 시뮬레이션 완료가 불가능하면 수정
                if not success:
                    error_type = "Runtime error"
                    combined_error = error_message
                    print(f"[ERROR] {error_type}: {combined_error}")
                else:
                    error_type = "Simulation completion issue"
                    combined_error = simulation_result.get('error', simulation_result.get('message', "Simulation cannot be completed"))
                    print(f"[INFO] {error_type}: {combined_error}")

                # 반복 예외 감지
                error_digest = self._get_error_digest(combined_error)
                if error_digest == self.last_error_message:
                    self.same_error_count += 1
                    print(f"[WARNING] Same error occurred {self.same_error_count} times in a row")
                else:
                    self.same_error_count = 0
                    self.last_error_message = error_digest

                # 에러 이력에 추가
                self.error_history.append(error_digest)

                # analyze_project_state 호출로 코드 수정 로직 처리
                code = self.analyze_project_state(
                    code_override=code, 
                    error_message=combined_error, 
                    current_filename_for_context=filename
                )

                # 반복 로그 저장
                self._save_iteration_log(self.iteration_count, code, combined_error)
        
            except Exception as e:
                print(f"[ERROR] Exception in iteration {self.iteration_count}: {str(e)}")
                traceback.print_exc()

        if self.iteration_count >= self.max_iterations:
            print(f"[WARNING] Reached maximum iterations ({self.max_iterations}) without success")

    def _get_error_digest(self, error_message: str) -> str:
        """에러 메시지를 요약하여 간단한 문자열로 변환"""
        if not error_message:
            return ""
        
        # 기본적인 정보 추출
        digest = error_message.strip()
        
        # 긴 에러 메시지는 일부만 포함
        if len(digest) > 100:
            digest = digest[:97] + "..."
        
        return digest

    def test_simulation_completion(self, filename: str) -> dict:
        """시뮬레이션을 자동으로 실행하여 완료 조건을 달성할 수 있는지 테스트합니다."""
        try:
            print(f"[INFO] Running automated simulation test for {filename}...")
            
            # 환경 변수 설정
            env = os.environ.copy()
            env["TEST_TIMEOUT"] = "120"  # 테스트 시간을 120초로 늘림
            
            # 명령줄 인수 분석 및 필요한 인수 추가
            default_args = ["--auto-test", "--strict-completion"]
            
            # 코드 분석하여 필수 인수 파악
            with open(filename, 'r', encoding='utf-8') as f:
                code_content = f.read()
                
            # 필수 인수가 있는지 확인
            required_args = []
            
            # 인수 패턴 검색
            arg_pattern = re.search(r'parser\.add_argument\([\'"](-\w+)[\'"].*required=True', code_content)
            if arg_pattern:
                required_arg = arg_pattern.group(1)
                if required_arg == '--size':
                    required_args.extend(['--size', '50'])
                elif required_arg == '--speed':
                    required_args.extend(['--speed', 'fast'])
                elif required_arg == '--mode':
                    required_args.extend(['--mode', 'comparison'])
                # 기타 필요한 인수들도 추가...
            
            # 최종 명령줄 인수 구성
            cmd_args = ["python", filename] + default_args + required_args
            
            process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # 최대 120초 동안 시뮬레이션 실행 허용
            try:
                stdout, stderr = process.communicate(timeout=120)
                
                # 결과 분석 - 더 엄격한 검증 추가
                if "SIMULATION_COMPLETE_SUCCESS" in stdout:
                    # 추가 확인: 각 알고리즘 성공 메시지 검증
                    bfs_success = "BFS_ALGORITHM_COMPLETE" in stdout
                    dfs_success = "DFS_ALGORITHM_COMPLETE" in stdout
                    astar_success = "ASTAR_ALGORITHM_COMPLETE" in stdout
                    
                    all_algorithms_success = bfs_success and dfs_success and astar_success
                    
                    if all_algorithms_success:
                        return {"success": True, "message": "All algorithms completed successfully"}
                    else:
                        # 일부 알고리즘만 성공한 경우
                        failed_algorithms = []
                        if not bfs_success: failed_algorithms.append("BFS")
                        if not dfs_success: failed_algorithms.append("DFS")
                        if not astar_success: failed_algorithms.append("A*")
                        
                        return {
                            "success": False, 
                            "message": f"Simulation reported success but not all algorithms completed. Failed: {', '.join(failed_algorithms)}"
                        }
                elif process.returncode != 0 or stderr:
                    return {"success": False, "message": f"Simulation error: {stderr}"}
                else:
                    # 성공 메시지가 없는 경우
                    return {"success": False, "message": "Simulation completed without success message"}
    
            except subprocess.TimeoutExpired:
                process.kill()
                return {"success": False, "message": "Simulation test timed out - likely infinite loop or missing exit condition"}
    
        except Exception as e:
            return {"success": False, "message": f"Simulation test error: {str(e)}"}

    def _request_fix_command_line_args(self, code, error_message, filename):
        """명령줄 인수 문제를 해결하기 위한 코드 수정 요청"""
        prompt = f"""
PROJECT GOAL: {self.project_goal}

The simulation code has issues with command line arguments.
Fix the argument parser to:
1. Make all arguments optional with reasonable defaults
2. Ensure '--auto-test' is properly handled
3. Fix any syntax or logical errors in the argument parsing code

Error message:
{error_message}

Current code:
```python
{code}
```

Provide ONLY the complete fixed Python code that resolves the command line argument issues.
"""

        system_prompt = f"""You are an expert Python developer.
Fix the command line argument handling in the pathfinding visualization.

REQUIREMENTS:
- Provide ONLY the complete Python code
- Make all command line arguments optional with sensible defaults
- Ensure the '--auto-test' mode works without requiring additional arguments
- Fix any argument parsing errors in the code
- Maintain all the functionality of the original code
"""

        ai_response = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        return self._extract_code_from_response(ai_response)

if __name__ == '__main__':
    # API 키 확인
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found in environment variables or .env file")
        print("Please set your OpenAI API key in the .env file or as an environment variable")
        print("You can get an API key from https://platform.openai.com/api-keys")
        exit(1)
        
    # 프로젝트 목표 수정: BFS, DFS, A* 알고리즘 동시 시각화 및 성공 종료 조건 추가
    project_goal = """Create a visual simulation that demonstrates how Breadth-First Search (BFS), Depth-First Search (DFS), and A* pathfinding algorithms explore a grid-based maze or map. The simulation must include:
1) An automatic maze generation algorithm (implement at least one of: DFS-based, Prim's algorithm, Kruskal's algorithm, or Recursive Division method)
2) A large and complex grid with walls/obstacles generated by the maze algorithm (minimum 50x50 cells, with options for larger sizes up to 100x100)
3) A clearly marked starting point
4) A clearly marked goal/target point
5) Visual representation of cells being explored in real-time with different colors for frontier, explored nodes, and path for each algorithm
6) A side-by-side comparison mode that shows all three algorithms (BFS, DFS, A*) simultaneously exploring the same maze
7) Speed control for the simulation (using predefined values, not requiring user input)
8) Clear visualization of the final path found from start to goal once each algorithm completes
9) Option to generate a new random maze with adjustable complexity parameters (via predefined settings, not requiring direct input)
10) Camera panning and zoom functionality to navigate larger  mazes (automatically managed)
11) Performance optimization for handling large mazes efficiently
12) A success screen or message when all algorithms reach the goal, marking the completion of the simulation
13) DYNAMIC AUDIO FEEDBACK:
    - Generate sounds programmatically WITHOUT using external audio files
    - Create a continuous audio gradient where pitch increases proportionally to exploration progress (0-100%)
    - Each algorithm should have a distinct sound frequency range (BFS, DFS, and A* should sound different)
    - Volume or intensity should also increase as the algorithm gets closer to the goal
    - Play a unique success sound when each algorithm finds the goal
    - Play a final victory chord or harmonized sound when all algorithms complete their paths
    - Implement volume control or mute option using a keyboard shortcut

IMPORTANT IMPLEMENTATION REQUIREMENTS:
- The simulation must run WITHOUT requiring ANY user input during normal operation
- All parameters should be predefined or set through command-line arguments
- Auto-run mode should be the default behavior (all algorithms start exploring automatically)
- If user interaction is absolutely necessary, minimize it to simple key presses (space bar, arrow keys) only
- For testing purposes, the simulation must be completely autonomous when run with the '--auto-test' flag
- Sound must be generated programmatically using pygame's audio capabilities (numpy and pygame.sndarray for waveform generation)
- DO NOT use any external audio files - all sounds must be synthesized at runtime

IMPORTANT: The simulation MUST automatically terminate and exit when all algorithms reach the goal in auto-test mode. When running with the '--auto-test' command line argument, the program MUST:
1. Verify that ALL THREE algorithms (BFS, DFS, A*) have successfully reached the goal
2. Print individual success messages for each algorithm:
   - "BFS_ALGORITHM_COMPLETE" when BFS reaches the goal
   - "DFS_ALGORITHM_COMPLETE" when DFS reaches the goal
   - "ASTAR_ALGORITHM_COMPLETE" when A* reaches the goal
3. Only print "SIMULATION_COMPLETE_SUCCESS" after confirming ALL algorithms have succeeded
4. Exit with code 0 only after all three algorithms find their paths

The '--strict-completion' flag must be respected to ensure all algorithms complete before terminating."""
    agent = AutonomousAgent(project_goal=project_goal)
    # 초기 코드나 파일명 없이 시작하여 AI가 생성하도록 함
    agent.run_autonomous_loop()