from __future__ import annotations
import json
import logging
import copy
from itertools import combinations
import pathlib
from typing import Callable, Tuple, List, Dict
from functools import partial
import os

from masif.models.baselines.lcdb_parametric_lc import ParametricLC
from masif.models.baselines.lcdb_parametric_best_lc import BestParametricLC
import numpy as np
import pysmt.fnode
from pysmt.shortcuts import (
    Symbol,
    And,
    GE,
    NotEquals,
    LT,
    Times,
    Pow,
    Minus,
    Plus,
    Exists,
    Real,
    Div,
    reset_env,
    Xor,
    Or,
    get_model,
    qelim,
    is_sat,
    ToReal,
    get_env,
)
from pysmt.typing import REAL
import inspect
from scipy.optimize import fsolve

logger = logging.getLogger(__name__)
x, y, z = [Symbol(s, REAL) for s in "xyz"]


def pow2(x: int, a: pysmt.fnode.FNode, b: float):
    return Times(Minus(Real(0.0), a), Real(x ** (-b)))


def pow3(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode, c: float):
    return Minus(a, Times(b, Real(x ** (-c))))


def log2(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode):
    return Plus(Times(Minus(Real(0.0), a), Real(np.log(x).item())), b)


def exp3(x: int, a: pysmt.fnode.FNode, b: float, c: pysmt.fnode.FNode):
    return Plus(Times(a, Real(np.exp(-b * x).item())), c)


def exp2(x: int, a: pysmt.fnode.FNode, b: float):
    return Times(a, Real(np.exp(-b * x).item()))


def lin2(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode):
    return Plus(Times(a, Real(x)), b)


# vap3 is not applicable for smt solvers


def mmf4(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode, c: pysmt.fnode.FNode, d: float):
    return Div(Plus(Times(a, b), Times(c, Real(x**d))), Plus(b, Real(x**d)))


def wbl4(x: int, a: float, b: pysmt.fnode.FNode, c: pysmt.fnode.FNode, d: float):
    return Minus(c, Times(b, Real(np.exp(-a * (x**d)).item())))


def exp4(x: int, a: float, b: float, c: pysmt.fnode.FNode, d: float):
    return Minus(c, Real(np.exp(-a * (x**d) + b).item()))


def expp3(x: int, a: float, b: float, c: pysmt.fnode.FNode):
    return Minus(c, Real(np.exp((x - b) ** a).item()))


def pow4(x: int, a: pysmt.fnode.FNode, b: pysmt.fnode.FNode, c: float, d: pysmt.fnode.FNode):
    return Minus(a, Times(b, Pow(Plus(Real(x), d), Real(c))))


def expd3(x: int, a: pysmt.fnode.FNode, b: float, c: pysmt.fnode.FNode):
    return Minus(c, Times(Minus(c, a), Real(np.exp(-b * x).item())))


def logpower3(x: int, a: pysmt.fnode.FNode, b: float, c: float):
    return Div(a, Real(1 + (x / np.exp(b).item()) ** c))


def last1(x: int, a: pysmt.fnode.FNode):
    return Minus(Plus(a, Real(x)), Real(x))


class SynthericParametericCurves(ParametricLC):
    def __init__(self, function: str, budgets: List[int], restarts: int = 10, parameters_lc: np.ndarray = np.ones(4)):
        super(SynthericParametericCurves, self).__init__(function, budgets, restarts)
        self.parameters_lc = [parameters_lc[: self.n_parameters]]
        if np.isnan(self.predict(self.budgets[0])):
            raise ValueError("Invalid Parameteric values")

    def fit(
        self,
        x: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        """
        :param parameters_lc: np.ndarray, a ndarray that
        """
        pass


def get_fixed_and_free_paras(func: Callable) -> Tuple[List, List]:
    """get the fixed and free parameters that can be passed to the SMT solver"""
    free_params = []
    fixed_params = []
    for key, value in inspect.signature(func).parameters.items():
        if key == "x":
            # 'x' is used to pass budget values
            continue
        if value.annotation == "pysmt.fnode.FNode":
            free_params.append(key)
        else:
            fixed_params.append(key)
    return free_params, fixed_params


class AbstractSyntheticParameterCurves(BestParametricLC):
    def __init__(self, budgets: List[int], restarts: int = 10, seed=0):
        super(AbstractSyntheticParameterCurves, self).__init__(budgets=budgets, restarts=restarts)
        self.restarts = restarts
        self.rng = np.random.RandomState(seed)

    def sample_lc(self, lc_type: str, para_init_values: dict, param_names: None | List[str] = None):
        """
        Sample the parameters of a learning curve. The parameters are pre-defined by a dict that store all the best-fit
        configurations from lcdb.

        : param lc_type: str, type of the learning curves
        : param para_init_values: dict, a dict recording the best fit parameters applied to the learning curve
        : param_names: names of the parameters that needs to be selected. If it set as None, we return all the parameters
        """
        all_parameters = para_init_values[lc_type]
        if np.isnan(all_parameters).any():
            all_parameters = np.asarray(all_parameters)[~np.isnan(all_parameters).any(axis=1)].tolist()

        lc_par = all_parameters[self.rng.randint((len(all_parameters)))]
        if param_names is None:
            return lc_par
        lc_par = {name: lc_par[ord(name) - ord("a")] for name in param_names}
        return lc_par

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.array([parametric_lc.predict(x) for parametric_lc in self.parametric_lcs]).squeeze()
        # fancy indexing to get the final prediction of the best (lowest cost during training) curve
        # for each algorithm.
        final_performance = predictions
        final_performance = final_performance + [(1 - np.arange(len(x)) / len(x))] * self.rng.normal(
            np.zeros_like(final_performance), 2 * np.ones_like(final_performance)
        )
        return final_performance

    def plot_curves(self, x, ax):
        y_hats = self.predict(x)

        for y_hat in y_hats:
            ax.plot(
                x,
                y_hat,
                color="red",
                alpha=0.5,
                linewidth=1.0,
            )
        ax.set_title("Generated Synthetic Functions")
        ax.set_xlabel("Budget")
        ax.set_ylabel("Performance")
        # plt.legend()
        plt.ylim(*(y_hats.min().item(), y_hats.max().item()))


class SyntheticParametericCurvesCalculus(AbstractSyntheticParameterCurves):
    functionals = ParametricLC.functionals

    def func_wrapper(
        self,
        input: np.ndarray,
        lc_type: str,
        budgets: list[int],
        target_values: list[float],
        n_func_pars: int,
        known_pars: dict[int, float],
    ):
        assert len(input) == len(budgets) == len(target_values)
        func = self.functionals[lc_type]
        func_pars = [0.0] * n_func_pars
        idx_input = 0
        for i in range(n_func_pars):
            if i in known_pars:
                func_pars[i] = known_pars[i]
            else:
                func_pars[i] = input[idx_input]
                idx_input += 1
        res = np.asarray([func(budget, *func_pars).item() - target for budget, target in zip(budgets, target_values)])
        return res.flatten()

    def fit(
        self,
        num_learning_curves: int,
        para_init_values: Dict,
        min_n_intersections: int = 5,
        intersection_budget_bounds: List[List[int]] = [[1, 5], [5, 10], [25, 50]],
    ) -> None:
        """
        Here I propose an incremental approach to generate a set of curves that has min_n_intersections intersection
        points:specifically, we first select the type of the parameters and find the number of their parameters. Then,
        given then number of parameters, we select the same amounf of anchor points to determine the parameters at hand.

        :param num_learning_curves: int, The number of learning curves that we wish to generate.
        :param para_init_values: Dict, a dictionary that indicate the proper exponent values for all the functions
            as they cannot be solved with the SMT solvers
        :param min_n_intersections: int the minimal number of intersection points
        :param intersection_budget_bounds: the bounds where the intersection needs to happen
        """
        curves_generated: List[SynthericParametericCurves] = []
        curves_types = []

        while len(curves_generated) < num_learning_curves:
            for idx_restart in range(self.restarts):

                new_lc_type = self.rng.choice(list(self.functionals.keys()), 1)[0]
                if len(curves_generated) == 0:
                    new_curve = SynthericParametericCurves(
                        new_lc_type,
                        budgets=self.budgets,
                        restarts=self.restarts,
                        parameters_lc=self.sample_lc(new_lc_type, para_init_values),
                    )
                    curves_generated.append(new_curve)
                    break
                else:

                    n_pars = len(inspect.signature(self.functionals[new_lc_type]).parameters) - 1  # x is not included

                    # There needs to be an intersection points between the selected curve and the new curve
                    curve = self.rng.choice(curves_generated, 1)[0]

                    x_start, x_end = intersection_budget_bounds[self.rng.randint(len(intersection_budget_bounds))]
                    x_start_is_smaller = 1 if self.rng.rand() < 0.5 else -1

                    y_curve_start = curve.predict(x_start)
                    y_curve_end = curve.predict(x_end)

                    # fixme: we might need a variance value that is larger for smaller budget values
                    #  and smaller for larger budget values
                    if y_curve_start > 0:
                        value_start = -1
                        while value_start < 0:
                            value_start = (1 - x_start_is_smaller * np.abs(self.rng.normal(0, 0.025))) * y_curve_start
                    elif y_curve_start < 0:
                        value_start = 1
                        while value_start > 0:
                            value_start = (1 - x_start_is_smaller * np.abs(self.rng.normal(0, 0.025))) * y_curve_start
                    else:
                        value_start = -x_start_is_smaller * np.abs(self.rng.normal(0, 0.025))

                    if y_curve_end > 0:
                        value_end = -1
                        while value_end < 0:
                            value_end = (1 + x_start_is_smaller * np.abs(self.rng.normal(0, 0.025))) * y_curve_end
                    elif y_curve_end < 0:
                        value_end = 1
                        while value_end > 0:
                            value_end = (1 + x_start_is_smaller * np.abs(self.rng.normal(0, 0.025))) * y_curve_end
                    else:
                        value_end = x_start_is_smaller * np.abs(self.rng.normal(0, 0.025))

                    diff = min(abs(value_start - y_curve_start), abs(value_end - y_curve_end))
                    if n_pars > 2:
                        n_remaining_pars = n_pars - 2

                        named_pars = ["a", "b", "c", "d"][:n_pars]
                        idx_remaining_pars = self.rng.choice(n_pars, n_remaining_pars, replace=False)
                        name_remaining_pars = [named_pars[i] for i in idx_remaining_pars]

                        lc_par_fixed = self.sample_lc(
                            lc_type=new_lc_type, para_init_values=para_init_values, param_names=name_remaining_pars
                        )
                        lc_par_fixed_ = {
                            idx: lc_par_fixed[name] for name, idx in zip(name_remaining_pars, idx_remaining_pars)
                        }
                    else:
                        lc_par_fixed_ = {}

                    func_wrapper = partial(
                        self.func_wrapper,
                        lc_type=new_lc_type,
                        budgets=[x_start, x_end],
                        target_values=[value_start, value_end],
                        n_func_pars=n_pars,
                        known_pars=lc_par_fixed_,
                    )
                    try:
                        root = fsolve(func_wrapper, np.asarray([0.0] * 2))
                        curve = [0.0] * n_pars
                        idx_root = 0
                        for i in range(n_pars):
                            if i in lc_par_fixed_.keys():
                                curve[i] = lc_par_fixed_[i]
                            else:
                                curve[i] = root[idx_root]
                                idx_root += 1

                        new_curve = SynthericParametericCurves(
                            new_lc_type, budgets=self.budgets, restarts=self.restarts, parameters_lc=np.asarray(curve)
                        )
                        res = new_curve.predict(self.budgets).flatten()
                        if np.isnan(res).any() or np.isinf(res).any():
                            continue
                        if np.max(res) > 100 or np.min(res) < 0:
                            continue
                        if res[-1] - res[0] < 20 or res[0] > 40:
                            continue

                        curves_generated.append(new_curve)
                        curves_types.append(new_lc_type)
                        break
                    except Exception as e:
                        print(e)
                        continue

            if idx_restart == self.restarts:
                # we randomly add a new parametric curve to the generated curve class
                lc_type = self.rng.choice(list(self.functionals.keys()), 1)
                new_curve = SynthericParametericCurves(
                    lc_type,
                    budgets=self.budgets,
                    restarts=self.restarts,
                    parameters_lc=self.sample_lc(lc_type, para_init_values),
                )
                curves_generated.append(new_curve)
                curves_types.append(lc_type)

            self.parametric_lcs = curves_generated
            self.curves_types = curves_types


class SyntheticParametericCurvesSMT(AbstractSyntheticParameterCurves):
    functionals = {
        "pow2": pow2,
        "pow3": pow3,
        "log2": log2,
        "exp3": exp3,
        "exp2": exp2,
        "lin2": lin2,
        # 'vap3': lambda x, a, b, c: np.exp(a + b / x + c * np.log(x)),
        "mmf4": mmf4,
        "wbl4": wbl4,
        "exp4": exp4,
        "expp3": expp3,
        # fun = lambda x: a * np.exp(-b*x) + c
        "pow4": pow4,  # has to closely match pow3,
        "expd3": expd3,
        "logpower3": logpower3,
        "last1": last1,
    }

    def fit(
        self,
        num_learning_curves: int,
        para_init_values: Dict,
        restarts: int = 10,
        min_n_intersections: int = 5,
        intersection_budget_bounds: List[List[int]] = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 50]],
    ) -> None:
        """
        Here I propose an incremental approach to generate a set of curves that has min_n_intersections intersection
        points: every time we pick 2 curves and check if they have an intersection points. One of these 2 curves must
        have free parameters. we try to generate a set of learning curves


        :param num_learning_curves: int, The number of learning curves that we wish to generate.
        :param para_init_values: Dict, a dictionary that indicate the proper exponent values for all the functions
            as they cannot be solved with the SMT solvers
        :param restarts: int, the number of restarts to use for the optimization.
        :param min_n_intersections: int the minimal number of intersection points
        :param intersection_budget_bounds: the bounds where the intersection needs to happen
        """
        n_generated_lcs = 0
        curves_generated: List[SynthericParametericCurves] = []
        curve_types = []

        n_curves_checked = 2  # This parameter indicates the amount of the learning curves that we want to compare at
        # each round

        while len(curves_generated) < num_learning_curves:
            for idx_restart in range(restarts):
                reset_env()
                get_env()
                all_fixed_lc = True
                while all_fixed_lc:
                    lc_indices = self.rng.choice(num_learning_curves, n_curves_checked, replace=False)
                    lc_indices.sort()
                    lc_is_generated = lc_indices < len(curves_generated)
                    n_curves_fixed = sum(lc_is_generated)
                    if n_curves_fixed < n_curves_checked:
                        all_fixed_lc = False

                x_start, x_end = intersection_budget_bounds[self.rng.randint(len(intersection_budget_bounds))]
                fixed_curves = [curves_generated[lc_idx] for lc_idx in lc_indices[:n_curves_fixed]]
                fixed_curve_bound_values = [[curve.predict(x_start), curve.predict(x_end)] for curve in fixed_curves]

                free_lcs_types = self.rng.choice(list(self.functionals.keys()), n_curves_checked - n_curves_fixed)
                lc_types, count_types = np.unique(free_lcs_types, return_counts=True)

                all_tested_lcs = []
                all_free_pars = []

                free_curve_bound_values = {}
                lcs_fixed_pars = {}
                lcs_free_pars = {}

                for lc_type, count_type in zip(lc_types, count_types):
                    free_params, fixed_params = get_fixed_and_free_paras(self.functionals[lc_type])

                    # fixme find a way to initialize fixed params
                    for idx in range(count_type):
                        lc_name = f"{lc_type}_{idx}"
                        all_tested_lcs.append(lc_name)

                        lc_fixed_par = self.sample_lc(
                            lc_type=lc_type, para_init_values=para_init_values, param_names=fixed_params
                        )
                        lc_free_par = {}
                        for free_par in free_params:
                            par = Symbol(f"{lc_type}_{idx}_{free_par}", REAL)
                            lc_free_par[free_par] = par
                            all_free_pars.append(par)

                        lcs_fixed_pars[lc_name] = lc_fixed_par
                        lcs_free_pars[lc_name] = lc_free_par

                        lc_value_start = self.functionals[lc_type](x=x_start, **lc_fixed_par, **lc_free_par)
                        lc_value_end = self.functionals[lc_type](x=x_end, **lc_fixed_par, **lc_free_par)
                        free_curve_bound_values[lc_name] = [lc_value_start, lc_value_end]

                smt_handles = []
                try:
                    # compare free-par curves with fixed_par_curves
                    for free_curve_bound in free_curve_bound_values.values():
                        for fix_bound in fixed_curve_bound_values:
                            smt_handles.append(
                                LT(
                                    Times(
                                        Minus(free_curve_bound[0], Real(fix_bound[0].item())),
                                        Minus(free_curve_bound[1], Real(fix_bound[1].item())),
                                    ),
                                    Real(0),
                                )
                                # Xor(
                                #    GE(free_curve_bound[0], Real(fix_bound[0].item())),
                                #    GE(free_curve_bound[1], Real(fix_bound[1].item())),
                                # )
                            )
                    for fre_curve_bound_1, fre_curve_bound_2 in combinations(free_curve_bound_values.values(), 2):
                        smt_handles.append(
                            LT(
                                Times(
                                    Minus(fre_curve_bound_1[0], fre_curve_bound_2[0]),
                                    Minus(fre_curve_bound_1[1], fre_curve_bound_2[1]),
                                ),
                                Real(0),
                            )
                            # Xor(
                            #   GE(fre_curve_bound_1[0], fre_curve_bound_2[0]),
                            #    GE(fre_curve_bound_1[1], fre_curve_bound_2[1]),
                            # )
                        )
                    problem = Or(*smt_handles) if len(smt_handles) > 1 else smt_handles[0]
                    domain = And(tuple(NotEquals(par, Real(0.0)) for par in all_free_pars))
                    f = And(domain, problem)

                    smt_model = get_model(f, solver_name="z3")
                except Exception as e:
                    print(e)
                    continue

                if smt_model:
                    for lc_name in all_tested_lcs:
                        lc_type = lc_name.split("_")[0]
                        lc_param = copy.deepcopy(lcs_fixed_pars[lc_name])
                        for key, value in lcs_free_pars[lc_name].items():
                            # cannot find how to properly transform the smt.node to float values.
                            # This is only a simple workaround...
                            lc_par = smt_model[value].serialize().split("/")

                            if len(lc_par) == 1:
                                lc_param[key] = float(lc_par[0])
                            else:
                                lc_param[key] = float(int(lc_par[0]) / int(lc_par[1]))
                        try:
                            new_curve = SynthericParametericCurves(
                                lc_type,
                                budgets=self.budgets,
                                restarts=restarts,
                                parameters_lc=np.asarray([lc_param[key] for key in sorted(lc_param.keys())]),
                            )
                            res = new_curve.predict(self.budgets).flatten()
                            if np.isnan(res).any() or np.isinf(res).any():
                                continue
                            if np.max(res) > 100 or np.min(res) < 0:
                                continue
                            if res[-1] - res[0] < 10:
                                continue
                        except Exception as e:
                            print(e)
                            continue

                        curves_generated.append(new_curve)
                        curve_types.append(curve_types)
                    break

            if idx_restart == restarts:
                # we randomly add a new parametric curve to the generated curve class
                lc_type = self.rng.choice(list(self.functionals.keys()), 1)
                new_curve = SynthericParametericCurves(
                    lc_type,
                    budgets=self.budgets,
                    restarts=restarts,
                    parameters_lc=self.sample_lc(lc_type, para_init_values),
                )
                curves_generated.append(new_curve)
                curve_types.append(lc_type)

        self.parametric_lcs = curves_generated
        self.curve_types = curve_types


class SyntheticFuncMetaDatasets(AbstractSyntheticParameterCurves):
    def fit(self, curves: List[SynthericParametericCurves]) -> None:
        """
        Generate the learning curves from a set of artifical functions by pertubabign their parameters
        """
        generated_curves = []
        for i, curve in enumerate(curves):
            lc_type = curve.function
            for idx_restart in range(self.restarts):
                lc_par = copy.deepcopy(curve.parameters_lc[0])

                new_value_idx = self.rng.randint(len(lc_par))

                new_value_coefficient = self.rng.normal(1, 0.2)
                while abs(new_value_coefficient - 1) > 0.3:
                    new_value_coefficient = self.rng.normal(1, 0.2)
                lc_par[new_value_idx] = lc_par[new_value_idx] * new_value_coefficient
                try:
                    new_curve = SynthericParametericCurves(
                        lc_type, budgets=self.budgets, restarts=curve.restarts, parameters_lc=lc_par
                    )
                except Exception as e:
                    print(e)
                    continue
                res = new_curve.predict(self.budgets).flatten()
                if np.isnan(res).any() or np.isinf(res).any():
                    continue
                if np.max(res) > 100 or np.min(res) < 0:
                    continue

                generated_curves.append(new_curve)
                break

            if idx_restart == self.restarts:
                print(i)
                generated_curves.append(curve)
        self.parametric_lcs = generated_curves


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with open(str(pathlib.Path(__file__).resolve().parent / "lcs_parameters.json"), "r") as f:
        para_init_values = json.load(f)
    budgets = list(range(1, 52))

    lc_predictor = SyntheticParametericCurvesCalculus(budgets, restarts=200, seed=0)
    lc_predictor.fit(50, para_init_values)
    lc_predictor.plot_curves(x=lc_predictor.budgets, ax=plt.gca())
    plt.show()

    new_curves_generator = SyntheticFuncMetaDatasets(budgets, restarts=200, seed=1)
    new_curves_generator.fit(lc_predictor.parametric_lcs)

    lcs_all = []
    lcs_all.append(lc_predictor.predict(lc_predictor.budgets))

    lcs_all.append(new_curves_generator.predict(lc_predictor.budgets))

    for i in range(100):
        new_curves_generator.fit(new_curves_generator.parametric_lcs)
        # new_curves_generator.plot_curves(x=lc_predictor.budgets, ax=plt.gca())
        # plt.show()
        lcs_all.append(new_curves_generator.predict(lc_predictor.budgets))

    # generated learning curves
    lcs_all = np.asarray(lcs_all)

    synthetic_func_path = (
        pathlib.Path(__file__).parent.parent.parent.parent.absolute() / "data" / "preprocessed" / "Synthetic"
    )

    if not synthetic_func_path.exists():
        os.makedirs(str(synthetic_func_path), exist_ok=True)

    with open(synthetic_func_path / "synthetic_func.npy", "wb") as f:
        np.save(f, lcs_all)
