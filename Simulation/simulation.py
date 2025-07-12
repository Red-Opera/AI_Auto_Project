from dotenv import load_dotenv
import os
import sys
import datetime
import subprocess
import openai
import re
import time

load_dotenv()

# 모델 설정 - OpenAI API 키와 모델 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # 기본 모델을 gpt-4.1-mini로 설정
TRANSLATION_MODEL = os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4.1-mini")  # 번역에도 동일 모델 사용

# AIInputText 관련 경로 설정
AI_INPUT_TEXT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AIInputText")
COMMON_TEXT_PATH = os.path.join(AI_INPUT_TEXT_DIR, "Common.txt")
PROJECT_GOAL_PATH = os.path.join(AI_INPUT_TEXT_DIR, "Simulation", "Simulation.txt")

# 보이드 시뮬레이션 설정 파일 경로
BOIDS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "boids-config.txt")

# 텍스트 파일 읽기 함수
def read_text_file(file_path):
    """파일 경로에서 텍스트 내용을 읽어옵니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"[ERROR] Failed to read file {file_path}: {str(e)}")
        return None

# 보이드 설정 파일 읽기 함수
def read_boids_config():
    """보이드 시뮬레이션 설정 파일을 읽어옵니다."""
    try:
        with open(BOIDS_CONFIG_PATH, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"[ERROR] Failed to read boids config file {BOIDS_CONFIG_PATH}: {str(e)}")
        return None

# 프로젝트 목표 및 공통 지침 로드
def load_project_texts():
    """AIInputText 폴더에서 프로젝트 목표와 공통 지침을 로드합니다."""
    project_goal = read_text_file(PROJECT_GOAL_PATH)
    common_guidelines = read_text_file(COMMON_TEXT_PATH)
    
    if not project_goal:
        print(f"[WARNING] Failed to load project goal from {PROJECT_GOAL_PATH}. Using default goal.")
        project_goal = "Create an advanced spatial hash-based Boids++ simulation system with flocking behavior, obstacle avoidance, and predator-prey interactions using Pygame visualization."
    
    if not common_guidelines:
        print(f"[WARNING] Failed to load common guidelines from {COMMON_TEXT_PATH}.")
    
    return project_goal, common_guidelines

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
            
        return True
    except Exception as e:
        print(f"[WARNING] Failed to save response to file: {e}")
        return False

# 명령어 실행 함수 - 30초 시간 제한 추가
def run_command(command, timeout=30):
    """명령어를 실행하되 30초 시간 제한을 적용합니다."""
    print(f"[INFO] Running command with {timeout}s timeout: {command}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    try:
        output, error = process.communicate(timeout=timeout)
        return output, error
    except subprocess.TimeoutExpired:
        print(f"[WARNING] Command timed out after {timeout} seconds, terminating process")
        
        # 프로세스 강제 종료
        try:
            # Windows에서는 taskkill 사용
            if os.name == 'nt':
                subprocess.run(f"taskkill /F /PID {process.pid} /T", shell=True, capture_output=True)
            else:
                # Unix/Linux에서는 SIGTERM 후 SIGKILL
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except Exception as kill_error:
            print(f"[ERROR] Failed to kill process: {kill_error}")
        
        # 타임아웃을 런타임 에러로 변환
        timeout_error = f"""TimeoutError: Program execution exceeded {timeout} seconds time limit.

PERFORMANCE ANALYSIS:
- The simulation failed to complete within the required {timeout}-second testing window
- This indicates potential performance issues or missing --auto-test implementation
- The program may be stuck in an infinite loop or running too slowly

REQUIRED FIXES FOR NEXT ITERATION:
1. Implement proper --auto-test mode that completes within {timeout} seconds
2. Add time management and automatic exit functionality
3. Optimize performance to meet timing requirements
4. Include progress indicators and time-based scenario switching
5. Ensure the program prints 'BOIDS_SIMULATION_COMPLETE_SUCCESS' and exits gracefully

SPECIFIC ISSUES TO ADDRESS:
- Missing or non-functional --auto-test command line argument handling
- Lack of time-limited testing scenarios
- Performance bottlenecks preventing timely completion
- Infinite loops or blocking operations without timeout handling
- Missing automatic exit mechanism after testing completion

ERROR TYPE: PerformanceError - Program execution timeout indicates optimization required for next iteration."""
        
        return "", timeout_error

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

# AI 호출 함수 - Common.txt 지침 포함
def call_ai(system_prompt: str, user_prompt: str, translate: bool = False, include_common_guidelines: bool = False) -> str:
    if translate:
        user_prompt = translate_to_english(user_prompt)
    
    try:
        # 공통 지침을 포함하는 경우
        if include_common_guidelines and common_guidelines:
            enhanced_system_prompt = f"{system_prompt}\n\n### Common Guidelines:\n{common_guidelines}"
        else:
            enhanced_system_prompt = system_prompt
            
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        # API 응답 내용
        response_content = response.choices[0].message.content
        
        # 전체 응답 객체 정보 저장 (프롬프트 전체 내용 포함)
        full_response_info = f"""
API Response Details:
---------------------
Model: {MODEL_NAME}
Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
System Prompt: {enhanced_system_prompt}
User Prompt: {user_prompt}

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

class AutonomousAgent:
    def __init__(self, project_goal: str = None, common_guidelines: str = None):
        self.max_iterations = 50
        self.project_root = os.getcwd()
        
        # AIInputText에서 로드한 텍스트 사용
        self.project_goal = project_goal if project_goal else "Create an advanced spatial hash-based Boids++ simulation system with flocking behavior, obstacle avoidance, and predator-prey interactions"
        self.common_guidelines = common_guidelines
        
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
        
        # 보이드 설정 파일 내용 로드
        self.boids_config_content = read_boids_config()
        
        # 같은 오류 발생 횟수 추적을 위한 딕셔너리
        self.error_occurrence_count = {}

    def _create_log_directory(self):
        """로그 디렉토리 생성"""
        try:
            os.makedirs(self.log_directory, exist_ok=True)
            print(f"[INFO] Log directory created: {self.log_directory}")
        except Exception as e:
            print(f"[ERROR] Failed to create log directory: {e}")

    def _save_iteration_log(self, iteration_number, code, error_message=None):
        """각 반복(Iteration)의 코드와 에러 메시지를 로그 파일에 저장"""
        try:
            log_filename = f"iteration_{iteration_number:03d}.log"
            log_path = os.path.join(self.log_directory, log_filename)
            
            with open(log_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"Iteration {iteration_number}\n")
                log_file.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write("=" * 50 + "\n\n")
                
                if error_message:
                    log_file.write("ERROR MESSAGE:\n")
                    log_file.write(error_message)
                    log_file.write("\n\n")
                
                log_file.write("CODE:\n")
                log_file.write(code)
                log_file.write("\n\n")
                
        except Exception as e:
            print(f"[ERROR] Failed to save iteration log: {e}")

    def _get_boids_config_reference(self):
        """보이드 설정 파일 참조 내용을 반환합니다."""
        if self.boids_config_content:
            return f"""
BOIDS CONFIGURATION REFERENCE:
==============================
File: boids-config.txt

Content:
{self.boids_config_content}

IMPORTANT: This file contains configuration parameters for the Boids++ simulation system.
Use these parameters to configure flocking behavior, spatial hash grid, obstacle avoidance,
and predator-prey interactions.
"""
        else:
            return "\n[WARNING] Boids configuration file could not be loaded.\n"

    def _should_include_boids_reference(self, error_message: str = None):
        """보이드 설정 파일 참조를 포함해야 하는지 판단합니다."""
        # 첫 번째 iteration에서는 항상 포함
        if self.iteration_count == 1:
            return True
            
        # 같은 오류가 3회 이상 발생한 경우
        if error_message:
            error_digest = self._get_error_digest(error_message)
            if error_digest in self.error_occurrence_count:
                if self.error_occurrence_count[error_digest] >= 3:
                    return True
                    
        return False

    def analyze_project_state(self, code_override: str = None, error_message: str = None, current_filename_for_context: str = None):
        generated_filename = None 

        if code_override:
            print(f"[INFO] Using provided code override for analysis")
            code = code_override
        else:
            print(f"[INFO] Analyzing current project state - iteration {self.iteration_count}")
            code = None

        # 오류 발생 횟수 추적
        if error_message:
            error_digest = self._get_error_digest(error_message)
            self.error_occurrence_count[error_digest] = self.error_occurrence_count.get(error_digest, 0) + 1
            print(f"[INFO] Error '{error_digest}' occurred {self.error_occurrence_count[error_digest]} times")

        # 보이드 설정 파일 참조가 필요한지 확인
        include_config_reference = self._should_include_boids_reference(error_message)
        config_reference = self._get_boids_config_reference() if include_config_reference else ""

        if code and error_message:
            # 에러가 있는 경우 부분 수정 또는 전체 재작성 결정
            if self._is_suitable_for_partial_fix(error_message):
                print("[INFO] Attempting partial fix...")
                try:
                    fixed_code = self._request_partial_fix(code, error_message, current_filename_for_context or "main.py")
                    if fixed_code:
                        return fixed_code, current_filename_for_context or "main.py"
                except Exception as e:
                    print(f"[ERROR] Partial fix failed: {e}")
                    
            print("[INFO] Attempting full code rewrite...")
            try:
                rewritten_code = self._request_full_rewrite(code, error_message, current_filename_for_context or "main.py", config_reference)
                return rewritten_code, current_filename_for_context or "main.py"
            except Exception as e:
                print(f"[ERROR] Full rewrite failed: {e}")
                
        # 새 코드 생성
        print("[INFO] Generating new code from scratch...")
        try:
            new_code_result = self._request_new_code_generation(config_reference)
            
            if isinstance(new_code_result, tuple) and len(new_code_result) == 2:
                return new_code_result
            else:
                # 단일 코드 반환인 경우
                return new_code_result, "boids_simulation.py"
                
        except Exception as e:
            print(f"[ERROR] Code generation failed: {e}")
            return None, None

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
            "process killed",
            "TimeoutError",
            "PerformanceError",
            "OptimizationRequiredError"
        ]
        
        error_lower = error_message.lower()
        
        # 전체 재작성이 필요한 경우
        if any(indicator in error_lower for indicator in full_rewrite_indicators):
            print(f"[INFO] Full rewrite needed due to: {error_message}")
            return False
            
        # 부분 수정 가능한 경우
        if any(indicator in error_message for indicator in partial_fix_indicators):
            print(f"[INFO] Partial fix possible for: {error_message}")
            return True
            
        # 기본적으로 부분 수정 시도
        return True

    def _extract_error_line_info(self, error_message: str) -> str:
        """에러 메시지에서 라인 정보 추출"""
        line_info = ""
        
        # "line X" 패턴 찾기
        line_match = re.search(r'line (\d+)', error_message)
        if line_match:
            line_info += f"Error occurred at line {line_match.group(1)}\n"
        
        # 파일명과 라인 정보가 함께 있는 경우
        file_line_match = re.search(r'File "([^"]+)", line (\d+)', error_message)
        if file_line_match:
            line_info += f"File: {file_line_match.group(1)}, Line: {file_line_match.group(2)}\n"
        
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

        system_prompt = f"""You are an expert Python and Pygame developer specializing in Boids simulation systems.
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
            return code_content
        else:
            return None

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
                return match.group(1).strip()
        
        # 코드 블록이 없으면 응답 전체에서 Python 코드 라인 찾기
        lines = ai_response.split('\n')
        code_lines = []
        in_code_section = False
        
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from ') or line.strip().startswith('def ') or line.strip().startswith('class '):
                in_code_section = True
            
            if in_code_section:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
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
            similarity = len(common_lines) / len(original_lines)
            if similarity < 0.7:  # 70% 이상 유사해야 부분 수정으로 간주
                return False
        
        # 문법 검증
        try:
            compile(fixed_code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            print(f"[ERROR] Syntax error in partial fix: {e}")
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

    def _request_full_rewrite(self, code: str, error_message: str, filename: str, config_reference: str = ""):
        """전체 코드 재작성 요청"""
        error_context_for_ai = ""
        if error_message:
            error_context_for_ai = f"\nPrevious error to fix: {error_message}\n"

        prompt = f"""
PROJECT GOAL: {self.project_goal}

{config_reference}

Fix the Python code for '{filename}' to resolve the error and achieve the project goal.
{error_context_for_ai}

Current code:
```python
{code}
```

Provide ONLY the complete corrected Python code in a single code block. No explanations.
"""
        system_prompt = f"""You are an expert Python developer specializing in Boids simulation systems. 
Fix the provided code to achieve: {self.project_goal}

REQUIREMENTS:
- Provide complete, working Python code
- Use a single ```python code block
- No explanations or text outside the code block
- Fix all errors and ensure the code runs properly

CRITICAL RULES FOR BOIDS++ SIMULATION DESIGN:
- Design a NEW, DIFFERENT boids behavior pattern for this iteration (iteration {self.iteration_count})
- Do not reuse the previous behavior patterns - create something fresh and unique
- Implement spatial hash grid for efficient collision detection and neighbor finding
- Use advanced flocking algorithms (separation, alignment, cohesion)
- Include obstacle avoidance and boundary handling
- Implement predator-prey interactions with different agent types
- Ensure smooth visualization with proper frame rate management
- Design intuitive controls for parameter adjustment
- Avoid infinite loops, performance bottlenecks, or unresponsive behavior
- Implement proper state management and real-time updates
- Include proper error handling for edge cases
- Optimize spatial hash grid cell size for performance

MANDATORY AUTOMATIC TESTING FEATURES:
1. When run with the '--auto-test' command line argument, the simulation must:
   - Run the boids simulation automatically for EXACTLY 30 seconds
   - Test all flocking behaviors and interactions within this time limit
   - Print 'BOIDS_SIMULATION_COMPLETE_SUCCESS' to stdout when testing is complete
   - Exit gracefully after 30 seconds or when testing is complete (whichever comes first)
   - Implement proper time management to ensure the test completes within 30 seconds
   - Use accelerated testing modes to demonstrate all features quickly
   - Include automatic scenario switching every 5-10 seconds to test different aspects
   - This mode is used to verify the simulation works correctly within the time constraint
   
TIMEOUT ERROR HANDLING:
- If previous error was a TimeoutError or PerformanceError, focus on optimization
- Implement proper --auto-test mode with time-limited execution
- Add automatic exit mechanisms and progress indicators
- Ensure all features can be tested within 30 seconds
- Include performance monitoring and FPS optimization
"""
    
        ai_response = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False
        )
        
        extracted_code = self._extract_code_from_response(ai_response)
        
        # 코드가 없거나 튜플인 경우 처리
        if not extracted_code:
            print("[WARNING] No code extracted from AI response")
            return None
        
        if isinstance(extracted_code, tuple):
            if len(extracted_code) == 2:
                return extracted_code[0]
                
        # 문자열이 아닌 경우 변환
        if not isinstance(extracted_code, str):
            extracted_code = str(extracted_code)
            
        return extracted_code

    def _request_new_code_generation(self, config_reference: str = ""):
        """새 코드 생성 요청 - Common.txt 지침 포함"""
        files = self.get_project_files()
        structure = self.get_directory_structure()
        
        prompt = f"""
PROJECT GOAL: {self.project_goal}

{config_reference}

Current project state:
Files: {files}
Directory structure: {structure}

Create complete Python code from scratch to achieve the project goal.

At the VERY BEGINNING of your response, include: # filename: chosen_filename.py
Then provide ONLY Python code in a code block, no explanations.
"""

        system_prompt = f"""You are an expert Python developer specializing in Boids simulation systems.
Create complete, working Python code from scratch to achieve: {self.project_goal}

REQUIREMENTS:
- Start with # filename: chosen_filename.py
- Provide complete, working Python code
- Use a single ```python code block
- No explanations or text outside the code block
- Ensure the code runs properly without errors

CRITICAL RULES FOR BOIDS++ SIMULATION:
- Implement spatial hash grid for efficient neighbor detection
- Create multiple agent types (boids, predators, obstacles)
- Implement advanced flocking behaviors (separation, alignment, cohesion)
- Add obstacle avoidance and boundary wrapping/bouncing
- Include predator-prey interactions with different behaviors
- Create smooth Pygame visualization with proper frame rate
- Implement real-time parameter adjustment controls
- Add spatial hash grid visualization for debugging
- Include performance metrics display
- Support the '--auto-test' command line argument functionality
- Print 'BOIDS_SIMULATION_COMPLETE_SUCCESS' when auto-test completes
- Optimize for smooth performance with many agents

MANDATORY 30-SECOND AUTO-TEST IMPLEMENTATION:
1. When '--auto-test' argument is provided:
   - Run simulation for EXACTLY 30 seconds maximum
   - Implement time tracking from start to finish
   - Automatically switch between different test scenarios every 5-10 seconds
   - Use accelerated simulation speed for comprehensive testing
   - Print progress updates during testing
   - Ensure graceful exit after 30 seconds or test completion
   - Test sequence should include: flocking (0-10s), predator-prey (10-20s), obstacles (20-30s)
   - Print 'BOIDS_SIMULATION_COMPLETE_SUCCESS' at the end
   - Exit the program automatically after printing the success message
   
PERFORMANCE REQUIREMENTS:
- Monitor FPS and ensure smooth performance
- Implement optimization strategies if performance drops
- Handle timeout scenarios gracefully
- Include proper error handling for all operations
"""
        
        ai_response = call_ai(
            system_prompt=system_prompt,
            user_prompt=prompt,
            translate=False,
            include_common_guidelines=True
        )
        
        # 파일명 추출
        filename_match = re.search(r'# filename: (.*\.py)', ai_response)
        if filename_match:
            filename = filename_match.group(1)
        else:
            filename = "boids_simulation.py"
            
        # 코드 추출
        code_content = self._extract_code_from_response(ai_response)
        
        if code_content:
            return code_content, filename
        else:
            return None, None

    def apply_code(self, code: str, filename: str):
        print(f"[INFO] Applying code to {filename}")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"[SUCCESS] Code applied to {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to apply code to {filename}: {e}")
            return False

    def test_code(self, filename: str):
        """코드 테스트 - 30초 시간 제한 적용"""
        print(f"[INFO] Testing code in {filename} with 30-second timeout")
        try:
            # 먼저 구문 검사
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()
            
            compile(code, filename, 'exec')
            print(f"[SUCCESS] Syntax check passed for {filename}")
            
            # 실행 테스트 - 30초 타임아웃 적용
            output, error = run_command(f"python {filename}", timeout=30)
            
            if error and ("TimeoutError" in error or "PerformanceError" in error):
                print(f"[ERROR] Performance/Timeout error in {filename}:")
                print(error)
                return False, error
            elif error:
                print(f"[ERROR] Runtime error in {filename}:")
                print(error)
                return False, error
            else:
                print(f"[SUCCESS] Code executed successfully: {filename}")
                print(f"Output: {output}")
                return True, output
                
        except SyntaxError as e:
            error_msg = f"Syntax error in {filename}: {e}"
            print(f"[ERROR] {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Error testing {filename}: {e}"
            print(f"[ERROR] {error_msg}")
            return False, error_msg

    def get_directory_structure(self):
        structure = []
        for root, dirs, files in os.walk(self.project_root):
            level = root.replace(self.project_root, '').count(os.sep)
            indent = ' ' * 2 * level
            structure.append(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if not file.startswith('.'):
                    structure.append(f"{subindent}{file}")
        return '\n'.join(structure)

    def get_project_files(self):
        files = []
        for root, dirs, filenames in os.walk(self.project_root):
            for filename in filenames:
                if filename.endswith('.py'):
                    files.append(os.path.join(root, filename))
        return files

    def run_autonomous_loop(self, initial_code: str = None, initial_filename: str = None):
        print(f"[INFO] Starting autonomous development loop")
        print(f"[INFO] Project goal: {self.project_goal}")
        print(f"[INFO] Max iterations: {self.max_iterations}")
        
        current_code = initial_code
        current_filename = initial_filename
        
        for iteration in range(1, self.max_iterations + 1):
            self.iteration_count = iteration
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # 코드 분석 및 생성
            code, filename = self.analyze_project_state(
                code_override=current_code,
                error_message=None,
                current_filename_for_context=current_filename
            )
            
            if not code:
                print(f"[ERROR] Failed to generate code in iteration {iteration}")
                continue
                
            # 코드 적용
            if self.apply_code(code, filename):
                # 로그 저장
                self._save_iteration_log(iteration, code)
                
                # 코드 테스트 (30초 제한 적용)
                success, result = self.test_code(filename)
                
                if success:
                    print(f"[SUCCESS] Iteration {iteration} completed successfully!")
                    
                    # 완료 테스트
                    completion_result = self.test_simulation_completion(filename)
                    if completion_result.get('success', False):
                        print(f"[SUCCESS] Project goal achieved in iteration {iteration}!")
                        print(f"[SUCCESS] Final file: {filename}")
                        return True
                    else:
                        print(f"[INFO] Code runs but project goal not yet achieved")
                        current_code = code
                        current_filename = filename
                else:
                    print(f"[ERROR] Iteration {iteration} failed with error:")
                    print(result)
                    
                    # 에러와 함께 다음 iteration에서 수정 시도
                    error_code, error_filename = self.analyze_project_state(
                        code_override=code,
                        error_message=result,
                        current_filename_for_context=filename
                    )
                    
                    if error_code:
                        current_code = error_code
                        current_filename = error_filename or filename
                    else:
                        print(f"[ERROR] Failed to generate error fix in iteration {iteration}")
                        
                    # 에러 로그 저장
                    self._save_iteration_log(iteration, code, result)
            else:
                print(f"[ERROR] Failed to apply code in iteration {iteration}")
                
        print(f"[WARNING] Maximum iterations ({self.max_iterations}) reached without success")
        return False

    def _get_error_digest(self, error_message: str) -> str:
        """에러 메시지의 핵심 부분을 추출하여 동일한 에러를 식별합니다."""
        # 에러 타입 추출
        error_type_match = re.search(r'(\w+Error)', error_message)
        if error_type_match:
            error_type = error_type_match.group(1)
        else:
            error_type = "UnknownError"
            
        # 에러 메시지의 첫 번째 줄 추출
        first_line = error_message.split('\n')[0].strip()
        
        # 파일 경로나 라인 번호 제거하여 일반화
        clean_line = re.sub(r'File "[^"]*", line \d+', '', first_line)
        clean_line = re.sub(r'line \d+', '', clean_line)
        
        return f"{error_type}:{clean_line[:100]}"  # 처음 100자만 사용

    # test_simulation_completion 메서드를 수정하여 점수를 확인하는 로직 추가
    def test_simulation_completion(self, filename: str) -> dict:
        """시뮬레이션이 완료되었는지 테스트합니다. 30초 시간 제한 포함."""
        try:
            # 30초 시간 제한으로 --auto-test 모드 실행
            print(f"[INFO] Starting 30-second auto-test for {filename}")
            
            # 30초 제한으로 테스트 실행
            start_time = time.time()
            output, error = run_command(f"python {filename} --auto-test", timeout=30)
            end_time = time.time()
            
            test_duration = end_time - start_time
            print(f"[INFO] Test completed in {test_duration:.2f} seconds")
            
            # 타임아웃/성능 에러 확인
            if error and ("TimeoutError" in error or "PerformanceError" in error):
                return {
                    'success': False,
                    'message': 'Test failed due to timeout or performance issues',
                    'output': output,
                    'error': error,
                    'timeout': True,
                    'duration': test_duration,
                    'score': 0
                }
            
            # 점수 확인 - "BOIDS_SIMULATION_SCORE: X/100" 패턴 찾기
            score_match = re.search(r'BOIDS_SIMULATION_SCORE:\s*(\d+)/100', output)
            if score_match:
                score = int(score_match.group(1))
                print(f"[INFO] Simulation score: {score}/100")
                
                # 100점 이상이면 성공으로 간주
                if score >= 100:
                    return {
                        'success': True,
                        'message': f'Boids simulation completed successfully with score {score}/100',
                        'output': output,
                        'duration': test_duration,
                        'timeout': False,
                        'score': score
                    }
                else:
                    return {
                        'success': False,
                        'message': f'Score {score}/100 is below required 100 points',
                        'output': output,
                        'error': f'Insufficient score: {score}/100',
                        'duration': test_duration,
                        'timeout': False,
                        'score': score
                    }
            
            # BOIDS_SIMULATION_COMPLETE_SUCCESS 메시지 확인
            if "BOIDS_SIMULATION_COMPLETE_SUCCESS" in output:
                return {
                    'success': True,
                    'message': 'Boids simulation completed successfully',
                    'output': output,
                    'duration': test_duration,
                    'timeout': False,
                    'score': 100  # 성공 메시지만 있고 점수가 없으면 기본 100점
                }
            else:
                return {
                    'success': False,
                    'message': 'Project goal not achieved',
                    'output': output,
                    'error': error,
                    'duration': test_duration,
                    'timeout': False,
                    'score': 0
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Test execution failed: {str(e)}',
                'error': str(e),
                'timeout': False,
                'duration': 0,
                'score': 0
            }

if __name__ == '__main__':
    # API key verification
    if not OPENAI_API_KEY:
        print("[ERROR] OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # AIInputText에서 프로젝트 목표 및 공통 지침 로드
    project_goal, common_guidelines = load_project_texts()
    
    # 프로젝트 목표 로드 확인
    if project_goal:
        print(f"[INFO] Project goal loaded: {project_goal[:100]}...")
    else:
        print("[WARNING] Failed to load project goal")
    
    # 에이전트 초기화 및 실행
    agent = AutonomousAgent(project_goal=project_goal, common_guidelines=common_guidelines)
    # Start without initial code or filename to let the AI generate it
    agent.run_autonomous_loop()