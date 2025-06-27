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
    def __init__(self, project_goal: str = "Create a Sokoban puzzle game"):
        self.max_iterations = 50
        self.project_root = os.getcwd()
        self.project_goal = project_goal
        self.iteration_count = 0
        
        # 로그 디렉토리 설정
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_directory = os.path.join(self.project_root, f"Log({self.timestamp})")
        self._create_log_directory()

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
        system_prompt = f"""You are an expert Python developer. 
Fix the provided code to achieve: {self.project_goal}

REQUIREMENTS:
- Provide complete, working Python code
- Use a single ```python code block
- No explanations or text outside the code block
- Fix all errors and ensure the code runs properly

CRITICAL RULES FOR GAME DESIGN:
- Design a NEW, DIFFERENT map layout for this iteration (iteration {self.iteration_count})
- Do not reuse the previous map design - create something fresh and unique
- Ensure all games have clear win and loss conditions
- Implement proper collision detection that works reliably
- Create balanced difficulty that provides appropriate challenge
- Design intuitive controls that respond consistently
- Avoid infinite loops, deadlocks, or unescapable situations
- Prevent impossible-to-win scenarios or unwinnable states
- Implement proper game state management and transitions
- Create reasonable AI behavior for computer-controlled elements
- Test all game mechanics to ensure they function as intended
- Include proper error handling for unexpected situations
- Ensure game performance is optimized for smooth gameplay

MANDATORY AUTOMATIC TESTING FEATURES:
1. When the environment variable TEST_COMPLETION=1 is set, the game must:
   - Implement an AI agent that can play through the game automatically
   - Use a predetermined solution path to complete the game
   - Print "GAME_COMPLETION_SUCCESS" to stdout when the game is won
   - Exit gracefully after completion
   - This mode is used to verify the game can actually be completed

2. Include a regular test mode activated with:
   - Command line argument: python game.py --test
   - Or environment variable: TEST_MODE=1
   - In this mode, initialize the game, show a frame, print "TEST_SUCCESS", then exit
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
        system_prompt = f"""You are an expert Python developer. 
Create code to achieve: {self.project_goal}

REQUIREMENTS:
- Start with: # filename: chosen_filename.py
- Provide complete Python code in ```python code blocks
- No explanations outside the code block
- Ensure the code achieves the project goal

CRITICAL RULES FOR GAME DESIGN:
- Design a NEW, UNIQUE map layout for this first iteration
- Create a creative and interesting level design
- Ensure all games have clear win and loss conditions
- Implement proper collision detection that works reliably
- Create balanced difficulty that provides appropriate challenge
- Design intuitive controls that respond consistently
- Avoid infinite loops, deadlocks, or unescapable situations
- Prevent impossible-to-win scenarios or unwinnable states
- Implement proper game state management and transitions
- Create reasonable AI behavior for computer-controlled elements
- Test all game mechanics to ensure they function as intended
- Include proper error handling for unexpected situations
- Ensure game performance is optimized for smooth gameplay

MANDATORY AUTOMATIC TESTING FEATURES:
1. When the environment variable TEST_COMPLETION=1 is set, the game must:
   - Implement an AI agent that can play through the game automatically
   - Use a predetermined solution path to complete the game
   - Print "GAME_COMPLETION_SUCCESS" to stdout when the game is won
   - Exit gracefully after completion
   - This mode is used to verify the game can actually be completed

2. Include a regular test mode activated with:
   - Command line argument: python game.py --test
   - Or environment variable: TEST_MODE=1
   - In this mode, initialize the game, show a frame, print "TEST_SUCCESS", then exit
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
            filename = "sokoban_pygame.py"  # 기본 파일명
        
        # 코드 추출
        extracted_code = self._extract_code_from_response(ai_response)
        
        # 문자열이 아닌 경우 처리
        if isinstance(extracted_code, tuple):
            if len(extracted_code) > 0:
                extracted_code = extracted_code[0]
            else:
                extracted_code = ""
            
        if not isinstance(extracted_code, str):
            extracted_code = str(extracted_code)
        
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
                # 테스트 모드 타임아웃은 무시 (게임이 예상대로 작동 중일 수 있음)
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
                    
                    # Pygame이 초기화되었다면 게임이 시작된 것으로 간주하고 성공으로 처리
                    return True, "Test PASSED: Pygame game started successfully. Timeout is expected for games with main loops."

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
                    final_message_parts.append("The timeout might be a consequence of the issue detailed above, or due to the game loop running as expected.")
                else: # 특정 에러 없이 타임아웃
                    final_message_parts.append("No specific Python traceback was captured before the timeout. The script might have an infinite loop, be waiting for input/resources, or the game loop is running as expected and exceeded the test duration.")
                
                # 코드 분석을 통한 추가 정보
                try:
                    # 게임 루프 분석
                    has_game_loop = any(pattern in current_code for pattern in ['while running:', 'while True:', 'while not done:'])
                    has_event_handling = 'pygame.event.get()' in current_code
                    has_quit_handling = 'pygame.QUIT' in current_code
                    has_draw_update = 'pygame.display.update()' in current_code or 'pygame.display.flip()' in current_code
                    
                    analysis_message = "\nCode Analysis:\n"
                    if has_game_loop and has_event_handling and has_quit_handling and has_draw_update:
                        analysis_message += "- Game appears to have proper game loop with event handling, quit mechanism, and display updates.\n"
                        analysis_message += "- Timeout likely indicates normal game execution rather than an error.\n"
                        analysis_message += "- This is EXPECTED behavior for a properly functioning game.\n"
                    else:
                        if not has_game_loop:
                            analysis_message += "- WARNING: No clear game loop detected. Game may exit prematurely or hang.\n"
                        if not has_event_handling:
                            analysis_message += "- WARNING: Event handling missing. Game may become unresponsive.\n"
                        if not has_quit_handling:
                            analysis_message += "- WARNING: No pygame.QUIT handling detected. Game may not exit properly.\n"
                        if not has_draw_update:
                            analysis_message += "- WARNING: No display update calls found. Game may not render properly.\n"
                    
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
        return 30 # 30초로 변경

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
                error_output = self.test_code(filename)
                
                # test_code가 튜플을 반환하는 경우 처리
                if isinstance(error_output, tuple):
                    stdout, stderr = error_output
                    error_message = stderr if stderr else stdout
                else:
                    error_message = error_output
                
                # 게임 자동 테스트 실행
                game_test_result = self.test_game_completion(filename)
                
                # 에러가 없고 게임을 클리어할 수 있으면 완료
                if not error_message and game_test_result.get('success', False):
                    print(f"[SUCCESS] Code passes all tests and game can be completed! Development finished.")
                    break
                
                # 에러가 있거나 게임 클리어가 불가능하면 수정
                error_type = "Runtime error" if error_message else "Game completion issue"
                combined_error = error_message or game_test_result.get('message', "Game cannot be completed")
                
                # 문자열이 아닌 경우 문자열로 변환
                if not isinstance(combined_error, str):
                    combined_error = str(combined_error)
                    
                print(f"[INFO] {error_type}: {combined_error}")
                
                # 맵 재생성이 필요한지 확인
                create_new_map = game_test_result.get('create_new_map', False)
                if create_new_map:
                    print(f"[INFO] Map design issue detected - requesting new map layout generation...")
                    # 맵 재생성을 위한 전체 코드 재작성 요청 (특별 플래그 추가)
                    code = self._request_full_rewrite(code, combined_error + " [GENERATE_NEW_MAP]", filename)
                else:
                    # 에러 타입에 따라 부분 수정 또는 전체 재작성
                    if error_message and isinstance(error_message, str) and self._is_suitable_for_partial_fix(error_message):
                        print(f"[INFO] Attempting partial fix...")
                        self._request_partial_fix(code, error_message, filename)
                    else:
                        print(f"[INFO] Requesting full code rewrite...")
                        code = self._request_full_rewrite(code, combined_error, filename)
                
                # 반복 로그 저장
                self._save_iteration_log(self.iteration_count, code, combined_error)
                
            except Exception as e:
                print(f"[ERROR] Exception in iteration {self.iteration_count}: {str(e)}")
                traceback.print_exc()
    
        if self.iteration_count >= self.max_iterations:
            print(f"[WARNING] Reached maximum iterations ({self.max_iterations}) without success")

    def test_game_completion(self, filename: str) -> dict:
        """게임을 자동으로 플레이하여 승리 조건을 달성할 수 있는지 테스트합니다."""
        try:
            print(f"[INFO] Running automated gameplay test for {filename}...")
            
            # 게임 프로세스 시작 (TEST_COMPLETION=1 환경변수로 테스트 모드 활성화)
            env = os.environ.copy()
            env["TEST_COMPLETION"] = "1"
            env["TEST_TIMEOUT"] = "60"  # 테스트 시간을 60초로 늘림
            
            process = subprocess.Popen(
                ["python", filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # 최대 60초 동안 게임 실행 허용 (테스트 시간 증가)
            try:
                stdout, stderr = process.communicate(timeout=60)
                
                # 결과 분석 - 더 철저한 메시지 검색
                if "GAME_COMPLETION_SUCCESS" in stdout:
                    return {"success": True, "message": "Game can be completed successfully"}
                elif "ERROR_BOXES_IN_PLACE_BUT_NO_WIN" in stdout:
                    return {"success": False, "message": "Game has a bug: Boxes are all in place but win condition not triggered", "create_new_map": False}
                elif "ERROR_UNSOLVABLE_LEVEL" in stdout:
                    return {"success": False, "message": "Game level is unsolvable - boxes cannot all be moved to targets", "create_new_map": True}
                elif "ERROR_DEADLOCK_DETECTED" in stdout:
                    return {"success": False, "message": "Game reached a deadlock state - puzzle cannot be completed", "create_new_map": True}
                elif "ERROR_NO_SOLUTION_FOUND" in stdout:
                    return {"success": False, "message": "No solution found after exhaustive search", "create_new_map": True}
                elif "ANALYSIS_UNSOLVABLE" in stdout:
                    return {"success": False, "message": "Map analysis indicates puzzle is unsolvable", "create_new_map": True}
                elif "TEST_TIMEOUT_WITHOUT_SOLUTION" in stdout:
                    return {"success": False, "message": "Test timed out without finding solution", "create_new_map": True}
                else:
                    # 솔루션을 찾지 못한 경우 기본적으로 맵을 변경
                    return {"success": False, "message": "Game completion test failed: " + stdout, "create_new_map": True}
        
            except subprocess.TimeoutExpired:
                process.kill()
                # 타임아웃도 맵 변경 트리거
                return {"success": False, "message": "Game completion test timed out - likely unsolvable", "create_new_map": True}
    
        except Exception as e:
            return {"success": False, "message": f"Game completion test error: {str(e)}", "create_new_map": False}

if __name__ == '__main__':
    # API 키 확인
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found in environment variables or .env file")
        print("Please set your OpenAI API key in the .env file or as an environment variable")
        print("You can get an API key from https://platform.openai.com/api-keys")
        exit(1)
        
    # 프로젝트 목표를 해결 가능한 난이도의 소코반 게임으로 변경
    project_goal = "Create a functional Sokoban puzzle game using pygame. The game should include a player character that can move around, boxes that can be pushed, target locations where boxes need to be placed, and walls. The game must use only basic geometric shapes (rectangles, circles) drawn with pygame.draw functions for all visual elements. Include basic collision detection, win condition checking when all boxes are on targets, and keyboard controls (arrow keys or WASD). No external image files or other asset files should be loaded or used. IMPORTANT: Design a complex map layout with at least 5 boxes and 5 target positions, multiple pathways, and challenging puzzle elements that require strategic thinking to solve. The map should offer a moderate challenge to players but MUST be solvable with proper strategy. Ensure the game is not impossibly difficult - a player should be able to solve the puzzle with reasonable effort and strategic thinking. Avoid any impossible configurations or deadlocks that make the game unwinnable."
    agent = AutonomousAgent(project_goal=project_goal)
    # 초기 코드나 파일명 없이 시작하여 AI가 생성하도록 함
    agent.run_autonomous_loop()
