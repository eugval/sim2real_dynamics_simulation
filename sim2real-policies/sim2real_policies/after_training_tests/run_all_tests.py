from sim2real_policies.final_policy_testing.pushing_test import main as pushing_main
from sim2real_policies.final_policy_testing.reaching_test import main as reaching_main
from sim2real_policies.final_policy_testing.sliding_test import main as sliding_main

if __name__ == "__main__":
    reaching_main()
    pushing_main()
    sliding_main()
