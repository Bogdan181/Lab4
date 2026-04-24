# Модель: Математична модель медичної діагностики (5 семестр)
# Автор: Ковальчук Богдан, група АІ-235

import numpy as np
from scipy.optimize import root, least_squares
import matplotlib.pyplot as plt


class DiagnosticModel:
    """
    Математична модель медичної діагностики за множиною біомаркерів.

    Система:
    y1 = 1 / (1 + exp(-(a * (a1*x1 + a2*x2) - S)))
    y2 = (beta1 * x3^n) / (beta2 + x3^n)
    y3 = k * ln(1 + gamma*x4) + delta * y2
    0  = y1 - tanh(y3 - tau)  (умова узгодженості)
    """

    def __init__(self, fixed_params, param_names_to_fit=None):
        self.params = fixed_params.copy()
        self.param_names_to_fit = param_names_to_fit or []

    def _equations(self, y, x, p):
        y1, y2, y3 = y
        x1, x2, x3, x4 = x

        logistic = 1.0 / (1.0 + np.exp(-(p['a'] * (p['a1'] * x1 + p['a2'] * x2) - p['S'])))
        eq1 = y1 - logistic

        x3_n = x3 ** p['n']
        eq2 = y2 - (p['beta1'] * x3_n) / (p['beta2'] + x3_n + 1e-9)

        eq3 = y3 - (p['k'] * np.log(1.0 + p['gamma'] * x4) + p['delta'] * y2)

        return np.array([eq1, eq2, eq3])

    def run_single(self, x_patient, params_override=None, y0=None):
        p = self.params.copy()
        if params_override is not None:
            p.update(params_override)

        if y0 is None:
            y0 = np.array([0.5, 0.5, 0.0])

        sol = root(self._equations, y0, args=(x_patient, p))
        y1, y2, y3 = sol.x

        residual_constraint = y1 - np.tanh(y3 - p['tau'])

        return {
            "y1": float(y1),
            "y2": float(y2),
            "y3": float(y3),
            "P_diagnosis": float(y1),
            "constraint_residual": float(residual_constraint),
            "success": bool(sol.success)
        }

    def objective_function(self, params_to_fit, x_data, y_target_data):
        p_local = self.params.copy()
        for name, val in zip(self.param_names_to_fit, params_to_fit):
            p_local[name] = val

        residuals = []
        for x, y_true in zip(x_data, y_target_data):
            res = self.run_single(x, params_override=p_local)
            y_pred = res["P_diagnosis"]
            constraint_res = res["constraint_residual"]

            residuals.append(y_pred - y_true)
            residuals.append(constraint_res)

        return np.array(residuals)

    def calibrate(self, x_data, y_target_data, initial_guess):
        result = least_squares(
            self.objective_function,
            x0=initial_guess,
            args=(x_data, y_target_data),
            method="trf"
        )

        for name, val in zip(self.param_names_to_fit, result.x):
            self.params[name] = float(val)

        return result

    def summary(self):
        print("=== Параметри моделі ===")
        for k, v in self.params.items():
            print(f"{k:7s} = {v:.4f}")
        print("========================")

if __name__ == "__main__":
    np.random.seed(42)

    true_params = {
        "a": 1.5, "a1": 0.8, "a2": 0.9, "beta1": 1.2, "beta2": 0.7,
        "n": 2, "k": 0.8, "gamma": 0.9, "delta": 0.5, "S": 0.2, "tau": 0.3
    }

    true_model = DiagnosticModel(true_params)

    N = 30
    X_data = 0.2 + 0.6 * np.random.rand(N, 4)

    y_true_list = []
    for x in X_data:
        r = true_model.run_single(x)
        y_true_list.append(r["P_diagnosis"])

    y_true = np.array(y_true_list)
    noise = 0.03 * np.random.randn(N)
    y_measured = np.clip(y_true + noise, 0.0, 1.0)

    initial_params = {
        "a": 1.5, "a1": 0.8, "a2": 0.9, "beta1": 1.0, "beta2": 1.0,
        "n": 2, "k": 0.8, "gamma": 0.7, "delta": 0.4, "S": 0.1, "tau": 0.3
    }

    params_to_fit = ["beta1", "beta2", "gamma", "delta", "S"]
    model = DiagnosticModel(initial_params, param_names_to_fit=params_to_fit)

    print("Параметри ДО калібрування:")
    model.summary()

    initial_guess = [model.params[name] for name in params_to_fit]
    result = model.calibrate(X_data, y_measured, initial_guess)

    print("\nРезультат калібрування (success =", result.success, ")")
    print("Кількість ітерацій:", result.nfev)
    print("Норма нев'язок:", np.linalg.norm(result.fun))

    print("\nПараметри ПІСЛЯ калібрування:")
    model.summary()

    x_norm = np.array([0.5, 0.5, 0.5, 0.5])

    x_inflam = x_norm.copy()
    x_inflam[2] = np.clip(x_inflam[2] * 1.4, 0.0, 1.0)

    x_metab = x_norm.copy()
    x_metab[3] = np.clip(x_metab[3] * 1.6, 0.0, 1.0)

    x_comb = x_norm.copy()
    x_comb[1] = np.clip(x_comb[1] * 1.3, 0.0, 1.0)
    x_comb[3] = np.clip(x_comb[3] * 1.5, 0.0, 1.0)

    scenario_names = [
        "Сценарій 1:\nнорма",
        "Сценарій 2:\n↑x3 (40%)",
        "Сценарій 3:\n↑x4 (60%)",
        "Сценарій 4:\n↑x2 & x4"
    ]
    scenario_x = [x_norm, x_inflam, x_metab, x_comb]

    P_scenarios = []
    for x in scenario_x:
        res = model.run_single(x)
        P_scenarios.append(res["P_diagnosis"])

    plt.figure()
    plt.bar(scenario_names, P_scenarios)
    plt.ylabel("P_діагноз")
    plt.title("Ймовірність діагнозу для різних сценаріїв")
    plt.ylim(0, 1)
    plt.grid(axis='y')

    x_base = x_norm.copy()

    a_values = np.linspace(1.0, 2.0, 40)
    tau_values = np.linspace(0.0, 0.6, 40)
    A, T = np.meshgrid(a_values, tau_values)
    P_heat = np.zeros_like(A)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            p_override = {"a": A[i, j], "tau": T[i, j]}
            res = model.run_single(x_base, params_override=p_override)
            P_heat[i, j] = res["P_diagnosis"]

    plt.figure()
    im = plt.imshow(
        P_heat,
        origin='lower',
        extent=[a_values.min(), a_values.max(), tau_values.min(), tau_values.max()],
        aspect='auto'
    )
    plt.colorbar(im, label="P_діагноз")
    plt.xlabel("a")
    plt.ylabel("tau")
    plt.title("Карта чутливості P_діагноз за параметрами a та τ")

    plt.tight_layout()
    plt.show()