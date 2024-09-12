import time
import os
import openai
import sys


class LMFunction(object):
    def __init__(self, engine='gpt-3.5-turbo', max_tokens=512):
        self.engine = engine
        self.max_tokens = max_tokens
        self.openai = openai
        # print("API Key e: ", os.environ['OPENAI_API_KEY'])
        openai.api_key = os.environ['OPENAI_API_KEY']

    # client = OpenAI()
    def _call_api(self, prompt, engine, max_tokens, max_retries=10, retry_wait=2):
        for i in range(max_retries):
            try:
                print("inside call api")
                return self.openai.chat.completions.create(
                    model=engine,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=1.0
                )
            except self.openai.OpenAIError as e:
                print('API call failed: %s' % e)
                time.sleep(retry_wait)
        return {'choices': [{'message': {'content': ''}}]}

    def _parse_message(self, msg):
        try:
            content = msg['choices'][0]['message']['content']
        except (IndexError, KeyError):
            content = ''
        return content

    def f(self, prompt, x):
        msg = self._call_api(
            prompt=prompt+x,
            engine=self.engine,
            max_tokens=self.max_tokens
        )
        evaluation = self._parse_message(msg)
        if evaluation == '':
            print('API call failed')
        return evaluation


class Checker(object):
    def __init__(self, working_dir, isa_path, theory_file, port=9000):
        sys.path.append(os.environ['PISA_PATH'])
        print(sys.path)
        try:
            from pisa_client import initialise_env
            self.initialise_env = initialise_env
            print("intialised_env")
        except:
            print("Set $PISA_PATH to /yourpath/to/Portal-to-ISAbelle/src/main/python")

        self.working_dir = working_dir
        self.isa_path = isa_path
        self.theory_file = theory_file
        self.port = port

    def _initialize(self):
        env = self.initialise_env(
            self.port,
            isa_path=self.isa_path,
            theory_file_path=self.theory_file,
            working_directory=self.working_dir
        )
        return env

    def _exit(self, env):
        try:
            env.post('exit')
        except:
            print("env.post('exit') timed out")
            pass
        os.system("ps aux | grep Isabelle | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
        os.system("ps aux | grep poly | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")

    def _parse_output(self, obs):
        """Parse the sledgehammer output, otherwise return an empty string"""
        if '<hammer>' in obs:
            output = obs.split('<hammer>')[0]
        else:
            output = ''
        return output

    def _run_step(self, step, i, tls_name, env):
        obs, reward, done, metadata = env.step_to_top_level_state(
            action=step,
            tls_name=tls_name,
            new_name='default_%d' % i
        )
        error = None
        if 'error:' in obs or 'Step error' in obs or 'Unknown error' in obs:
            error = obs
        return obs, reward, done, metadata, error

    def _run_sledgehammer(self, step, i, tls_name, env):
        # First try heuristics
        #TODO : put sos back in 
        for heuristic in ['by auto', 'by sos', 'by simp', 'by blast', 'by fastforce', 'by force', 'by eval', 'by presburger', 'by arith', 'by linarith', 'by (auto simp: field_simps)']:
            step_ = step.replace('normalhammer', heuristic)
            print("heuristic: ", heuristic)
            obs, reward, done, metadata, error = self._run_step(step_, i, tls_name, env)
            print("error: ", error)
            if error is None:
                obs = '%s <hammer> %s' % (heuristic, obs)
                return obs, reward, done, metadata, error
        # Try sledgehammer
        out = self._run_step(step, i, tls_name, env)
        return out

    def check(self, statement_and_proof):
        # Initialize environment
        print("in check")
        env = self._initialize()
        env.initialise()

        # Wrap and parse theorem
        theory = Checker.wrap_theorem(statement_and_proof)
        steps = Checker.get_parsed(env, theory)


        # steps = env.post(f"<parse text> ${theory}")
        # steps = steps.split('<SEP>')
        # steps = [s for s in steps if s.strip() != '']
        # # remove weird '$' step and whitespace steps
        # steps = [s for s in steps if s != '$' and s.strip() != '']

        print("theory: ", theory)
        print("-----------------")
        print("steps: ", steps)

        result = self._check(env, steps)
        return result

    def _check(self, env, steps):
        done = False
        reason = ''
        success = False
        step_results = []
        tls_name = 'default'
        for i, step in enumerate(steps):
            try:
                time0 = time.time()
                if 'normalhammer' in step:
                    obs, reward, done, metadata, error = self._run_sledgehammer(step, i, tls_name, env)
                else:
                    obs, reward, done, metadata, error = self._run_step(step, i, tls_name, env)
                # obs, reward, done, metadata, error = self._run_step(step, i, tls_name, env)
                step_time = time.time() - time0
                step_results.append(dict(index=i, step=step, output=self._parse_output(obs), step_time=step_time))
                if error is not None:
                    reason = error
                    success = False
                    done = False
                    break
            except:
                # Timeout - end the proof attempt
                success = False
                done = False
                reason = 'timeout (%d)' % len(step_results)
                step_results.append(dict(index=i, step=step, output=''))
                break

            # Change when successful
            tls_name = 'default_%d' % i

        if done and reward == 1.0:
            success = True

        result = {
            'success': success,
            'reason': reason,
            'num_steps': len(steps),
            'last_step': len(step_results),
            'step_results': step_results,
            'theorem_and_proof': self.reconstruct(step_results) if success else ''
        }
        # Exit environment
        self._exit(env)
        return result
    
    @staticmethod
    def reconstruct(step_results):
        steps = []
        for step_result in step_results[1:]:
            if step_result['output'] != '':
                steps.append(step_result['output'].strip())
            else:
                steps.append(step_result['step'].strip())
        theorem_and_proof = '\n'.join(steps)
        return theorem_and_proof

    @staticmethod
    def wrap_theorem(theorem):
        return 'theory Interactive imports HOL.HOL Complex_Main "HOL-Library.Code_Target_Numeral" "HOL-Library.Sum_of_Squares" "Symmetric_Polynomials.Vieta" "HOL-Computational_Algebra.Computational_Algebra" "HOL-Number_Theory.Number_Theory" "HOL-Decision_Procs.Approximation" \n begin\n%s' % theorem

    @staticmethod
    def get_parsed(env, theory, tls_name='default'):
        # HACK: the parsing doesn't work well with `normalhammer`, so we replace
        # all hammer calls with sorry, then replace sorry to normalhammer after parsing.
        print("in get_parsed")
        theory = theory.replace('sledgehammer', 'sorry')
        theory = theory.replace('normalhammer', 'sorry')

        steps = env.post(f"<parse text> ${theory}")
        steps = steps.split('<SEP>')
        steps = [s for s in steps if s.strip() != '']
        # remove weird '$' step and whitespace steps
        steps = [s for s in steps if s != '$' and s.strip() != '']
        steps = [s.replace('sorry', 'normalhammer') for s in steps]
        return steps
