**DRL-Based Medium-Term Planning of Renewable-Integrated Self-Scheduling Cascaded Hydropower to Guide Wholesale Market Participation**

Welcome to the repository accompanying our paper, *"DRL-Based Medium-Term Planning of Renewable-Integrated Self-Scheduling Cascaded Hydropower to Guide Wholesale Market Participation."*

> **Important Note:**  
> Due to collaboration policies, we are unable to provide data related to the PGE system. Nevertheless, we made an effort to upload as many files as possible to showcase the DRL portion of our research.

---

## Main Repository Contents

- **CR_Result.mat**  
  Contains the results obtained using the multi-parametric programming acceleration method.

- **env.py**  
  Defines the environment using the [Gymnasium](https://gymnasium.farama.org/) framework.

- **lib_operation_model.py**  
  Provides the operating model libraries, including the scheduling model used in our paper.

- **scenarios_1500.csv**  
  Includes stochastic scenarios for variable renewable energy sources.

- **test_SAC.py**  
  Shows how we tested the Soft Actor-Critic (SAC) algorithm in our study.

- **train_SAC.py**  
  Demonstrates how we trained the SAC algorithm for medium-term planning of hydropower operations.
