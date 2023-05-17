import flwr.server.strategy as st

strategies = {
    'fedavg': st.FedAvg,
    'fedavgm': st.FedAvgM,
    'qfedavg': st.QFedAvg,
    'ftfedavg': st.FaultTolerantFedAvg,
    'fedopt': st.FedOpt,
    'fedprox': st.FedProx,
    'fedadagrad': st.FedAdagrad,
    'fedadam': st.FedAdam,
    'fedyogi': st.FedYogi, 
}

def get_strategy(strategy='FedAvg', config={}):
    return strategies[strategy.lower()](**config)

