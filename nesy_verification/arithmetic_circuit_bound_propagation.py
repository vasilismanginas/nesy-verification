from pysdd.sdd import SddNode
from functools import reduce
from pysdd.sdd import SddManager

import graphviz
import operator
import gurobipy as gp
from torch.autograd import variable


def sdd_to_gurobi_model(
    node: SddNode,
    bounds: dict[int, tuple[float, float]],
    categorical_groups: list[list[int]],
):

    # create environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model, categorical_values = (
        gp.Model(env=env),
        set(value for group in categorical_groups for value in group),
    )

    variable_vars = {
        key: model.addVar(
            lb=lb, ub=ub, vtype=gp.GRB.CONTINUOUS, name="literal:{}".format(key)
        )
        for key, (lb, ub) in bounds.items()
    }

    for group in categorical_groups:
        model.addConstr(
            reduce(operator.add, [variable_vars[var] for var in group]) == 1
        )

    def depth_first_search(node: SddNode):
        if node.is_true():
            # print("adding true leaf node {}".format(node.id))
            return 1
        elif node.is_false():
            # print("adding false leaf node {}".format(node.id))
            return 0
        elif node.is_literal():
            literal_id = abs(node.literal)
            if node.literal > 0:
                var = model.addVar(
                    vtype=gp.GRB.CONTINUOUS, name="node:{}".format(node.id)
                )
                model.addConstr(var == variable_vars[literal_id])
                # print("added positive literal node {}".format(node.literal))
                return var
            else:
                # Negative literals of categorical variables always get 1
                if literal_id in categorical_values:
                    # print(
                    #     "added negative literal node {} with value 1 (categorical)".format(
                    #         node.literal
                    #     )
                    # )
                    return 1

                var = model.addVar(
                    vtype=gp.GRB.CONTINUOUS, name="node:{}".format(node.id)
                )
                model.addConstr(var == 1 - variable_vars[literal_id])
                # print("added negative literal node {} (binary)".format(node.literal))
                return var

        elif node.is_decision():
            var = model.addVar(vtype=gp.GRB.CONTINUOUS, name="var_{}".format(node.id))
            model.addConstr(
                var
                == reduce(
                    operator.add,
                    [
                        operator.mul(depth_first_search(prime), depth_first_search(sub))
                        for prime, sub in node.elements()
                    ],
                )
            )
            # print("added decision node: {}".format(node.id))
            return var
        else:
            raise RuntimeError("what is happening?")

    optimization_target = depth_first_search(node)

    return model, optimization_target


def get_expression_min_max(
    transition: str, variable_bounds: dict[int, tuple[float, float]]
):
    # declare the variables that we need
    manager = SddManager(4, 0)
    (
        smaller_than_3,
        between_3_and_6,
        larger_than_6,
        even,
    ) = [manager.literal(i) for i in range(1, 5)]

    # these constraints state that these 3 variables are mutually exclusive
    # i.e. one and only one of these must happen
    constraints = (
        (smaller_than_3 | between_3_and_6 | larger_than_6)
        & (~smaller_than_3 | ~between_3_and_6)
        & (~smaller_than_3 | ~larger_than_6)
        & (~between_3_and_6 | ~larger_than_6)
    )

    if transition == "t1":
        expression = even & larger_than_6 & constraints
    elif transition == "t2":
        expression = (~even) & (~larger_than_6) & constraints
    elif transition == "t3":
        expression = smaller_than_3 & constraints
    else:
        raise (ValueError("'transition' must be 't1', 't2', or 't3'"))

    model, var = sdd_to_gurobi_model(
        expression,
        bounds=variable_bounds,
        categorical_groups=[[1, 2, 3]],
    )

    # find minimum
    model.setObjective(var, sense=gp.GRB.MINIMIZE)
    model.optimize()
    minimum = model.ObjVal

    # find maximum
    model.setObjective(var, sense=gp.GRB.MAXIMIZE)
    model.optimize()
    maximum = model.ObjVal

    return minimum, maximum
