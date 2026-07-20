from grid_based.custom_environment_p import GridWithMemory
import pettingzoo
from pettingzoo.test import parallel_api_test
import numpy.random as npr
import logging, multiprocessing, time
from math import trunc
if __name__ == "__main__" :

    logger = logging.getLogger(__name__)

    gridworld = (GridWithMemory())
    gridworld.reset()
    #gridworld.render()
    def random_test(num_cycles:int,log_name:str) -> None :
        gridworld = (GridWithMemory())
        logging.basicConfig(filename=f"{log_name}.log", level=logging.INFO)
        gridworld.reset()
        for i in range(num_cycles) :
            gridworld.reset()
            for j in range (100) :
                action1=npr.randint(low=0, high=4)
                action2=npr.randint(low=0, high=4)
                action3=npr.randint(low=0, high=4)


                gridworld.step(actions={"agent_0":action1,"agent_1":action2,"agent_2":action3})
                #gridworld.render()
                #logger.info(gridworld.render())

    num_threads=1
    threads = []

    for i in range(num_threads) :
        t=multiprocessing.Process(target=random_test, kwargs={"num_cycles":1,"log_name":"logname"})
        threads.append(t)
    #print(threads)


    time_one = time.time()
    for t in threads:
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()
    time_two = time.time()
    print(f"time of {num_threads} thread for 1_000_000 (8*125_000) ops is {(trunc(time_two-time_one))} secs")
