import os
import socket
import subprocess
import time
import threading
from dsp_functions import *


class Prover:
    def __init__(self) -> None:
        os.environ["PISA_PATH"] = (
            "/local/scratch/dc755/Portal-to-ISAbelle/src/main/python"
        )
        self.checker = Checker(
            working_dir="/local/scratch/dc755/Isabelle2022/src/HOL/Examples",
            isa_path="/local/scratch/dc755/Isabelle2022",
            theory_file="/local/scratch/dc755/Isabelle2022/src/HOL/Examples/Interactive.thy",
            port=8000,
        )

    def start_server(self):
        directory = "/local/scratch/dc755/Portal-to-ISAbelle/"
        command = 'sbt "runMain pisa.server.PisaOneStageServer8000"'
        subprocess.run(command, shell=True, cwd=directory, text=True)

        

    def is_port_in_use(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.settimeout(1)
                sock.connect((host, port))
                return True
            except (socket.timeout, ConnectionRefusedError):
                return False

    def check_proof(self, proof: str) -> bool:
        # print(self.is_port_in_use(host="127.0.0.1", port=8000))
        return self.checker.check(proof)


if __name__ == "__main__":
    prover = Prover()
    code = r"""
lemma gcd_diff1:
  fixes n :: nat
  shows "gcd (21 * n + 4) (14 * n + 3) = gcd ((21 * n + 4) - (14 * n + 3)) (14 * n + 3)"
  \<proof>

lemma gcd_diff2:
  fixes n :: nat
  shows "gcd (7 * n + 1) (14 * n + 3) = gcd (7 * n + 1) ((14 * n + 3) - (7 * n + 1))"
proof -
  have "gcd (7 * n + 1) (14 * n + 3) = gcd ((14 * n + 3) - (7 * n + 1)) (7 * n + 1)"
    using gcd_red_nat
  by (smt (verit) add.assoc add.commute add.left_commute diff_add_inverse gcd.commute left_add_mult_distrib mod_add_self1 numeral_Bit0 numeral_Bit1)
  also have "... = gcd (7 * n + 2) (7 * n + 1)"
    by (simp)
  finally show ?thesis 
    by simp
qed

lemma gcd_diff3:
  fixes n :: nat
  shows "gcd (7 * n + 1) (7 * n + 2) = gcd (7 * n + 1) ((7 * n + 2) - (7 * n + 1))"
proof -
  have "gcd (7 * n + 1) (7 * n + 2) = gcd ((7 * n + 2) - (7 * n + 1)) (7 * n + 1)"
    using gcd_red_nat by simp
  also have "... = gcd 1 (7 * n + 1)"
    by (simp)
  finally show ?thesis
    by auto
qed

theorem imo_1959_p1:
  fixes n :: nat
  shows "gcd (21 * n + 4) (14 * n + 3) = 1"
proof -
  have "gcd (21 * n + 4) (14 * n + 3) = gcd (7 * n + 1) (14 * n + 3)"
    using gcd_diff1 by simp
  also have "... = gcd (7 * n + 1) (7 * n + 2)"
    using gcd_diff2 by simp
  also have "... = gcd (7 * n + 1) 1"
    using gcd_diff3 by simp
  also have "... = 1"
    by simp
  finally show ?thesis .
qed
"""
    result = prover.check_proof(code)
    print(f"####### Success: {result['success']} ########")
    print("##### output ########")

    print(result["theorem_and_proof"])
