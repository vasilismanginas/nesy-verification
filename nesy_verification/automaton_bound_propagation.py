import gurobipy as gp
from gurobipy import GRB


def get_state_min_max(
    state_probs: dict[str, list[float]],
    symbol_probs: dict[str, list[float]],
    state_expression: str,
):
    try:
        # create environment and model
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)

        # Add constraints for symbol and state lower and upper bounds
        for symbol in symbol_probs.keys():
            globals()[symbol] = m.addVar(vtype=GRB.CONTINUOUS, name=symbol)
            m.addConstr(globals()[symbol] >= symbol_probs[symbol][0], f"{symbol}_lower")
            m.addConstr(globals()[symbol] <= symbol_probs[symbol][1], f"{symbol}_upper")

        for state in state_probs.keys():
            globals()[state] = m.addVar(vtype=GRB.CONTINUOUS, name=state)
            m.addConstr(globals()[state] >= state_probs[state][0], f"{state}_lower")
            m.addConstr(globals()[state] <= state_probs[state][1], f"{state}_upper")

        # Add constraints for symbol and state probability distributions
        m.addConstr(sum(globals()[state] for state in state_probs.keys()) == 1, "c2")  # type: ignore

        # find minimum
        m.setObjective(eval(state_expression), GRB.MINIMIZE)
        m.optimize()
        minimum = m.ObjVal

        # find maximum
        m.setObjective(eval(state_expression), GRB.MAXIMIZE)
        m.optimize()
        maximum = m.ObjVal

        return [minimum, maximum]

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

    except Exception as e:
        print(e)
