"""
FraudSight Engine - Phase 2 & 3 Detection Modules
Handles: Device fingerprint risk, velocity detection, beneficiary alerts,
unusual hour detection, Random Forest ML, and SHAP explainability.
"""

import hashlib
import math
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def get_transaction_hour(step: int) -> int:
    """Convert simulation step (hours) to hour-of-day (0-23)."""
    return step % 24


def is_unusual_hour(step: int) -> tuple[bool, str]:
    """Flag transactions during late-night / early-morning hours."""
    hour = get_transaction_hour(step)
    if 0 <= hour < 5:
        return True, f"Late-night transfer at {hour:02d}:00 hrs"
    elif 22 <= hour < 24:
        return True, f"Late-night transfer at {hour:02d}:00 hrs"
    return False, ""


# ─────────────────────────────────────────────
# DEVICE FINGERPRINT RISK
# ─────────────────────────────────────────────

KNOWN_DEVICES = {}  # account -> list of device fingerprints


def generate_device_fingerprint(account_id: str, amount: float, step: int) -> str:
    """
    Simulate a device fingerprint from transaction metadata.
    In production, this would come from browser/app telemetry.
    """
    seed = f"{account_id}-{int(amount) // 1000}-{step // 72}"
    return hashlib.md5(seed.encode()).hexdigest()[:12]


def assess_device_risk(account_id: str, fingerprint: str) -> tuple[float, str]:
    """
    Returns (risk_score 0-1, reason).
    New device on high-value account = higher risk.
    """
    if account_id not in KNOWN_DEVICES:
        KNOWN_DEVICES[account_id] = []

    known = KNOWN_DEVICES[account_id]

    if len(known) == 0:
        KNOWN_DEVICES[account_id].append(fingerprint)
        return 0.1, "First transaction (no device history)"

    if fingerprint not in known:
        risk = min(0.4 + 0.1 * len(known), 0.85)
        KNOWN_DEVICES[account_id].append(fingerprint)
        return risk, f"New device fingerprint (seen {len(known)} previous devices)"

    return 0.05, "Known device"


def reset_device_registry():
    global KNOWN_DEVICES
    KNOWN_DEVICES = {}


# ─────────────────────────────────────────────
# TRANSACTION VELOCITY DETECTION
# ─────────────────────────────────────────────

class VelocityTracker:
    """
    Tracks transactions per account within rolling time windows.
    Uses step-based time (1 step = 1 hour in the dataset).
    """

    def __init__(self, window_hours: int = 2, threshold: int = 5):
        self.window = window_hours  # hours
        self.threshold = threshold
        self.history: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def record_and_check(self, account_id: str, step: int, amount: float) -> tuple[bool, str, int]:
        """
        Records transaction and returns (is_flagged, reason, count_in_window).
        """
        # Prune old entries outside the window
        cutoff = step - self.window
        self.history[account_id] = [
            (s, a) for s, a in self.history[account_id] if s >= cutoff
        ]

        # Record this transaction
        self.history[account_id].append((step, amount))
        count = len(self.history[account_id])

        if count >= self.threshold:
            return (
                True,
                f"{count} transactions within {self.window} hours (threshold: {self.threshold})",
                count
            )
        return False, "", count

    def get_velocity(self, account_id: str, step: int) -> int:
        cutoff = step - self.window
        return sum(1 for s, _ in self.history[account_id] if s >= cutoff)

    def reset(self):
        self.history = defaultdict(list)


# ─────────────────────────────────────────────
# FIRST-TIME BENEFICIARY ALERT
# ─────────────────────────────────────────────

class BeneficiaryTracker:
    """
    Tracks sender→recipient relationships.
    Flags first-time transfers to new beneficiaries.
    """

    def __init__(self):
        self.relationships: dict[str, set] = defaultdict(set)

    def check_and_register(self, sender: str, recipient: str, amount: float) -> tuple[bool, str]:
        """Returns (is_new_beneficiary, reason)."""
        is_new = recipient not in self.relationships[sender]

        if is_new:
            self.relationships[sender].add(recipient)
            known_count = len(self.relationships[sender]) - 1
            reason = (
                f"First-time transfer to {recipient}"
                if known_count == 0
                else f"New beneficiary {recipient} (you have {known_count} known recipients)"
            )
            return True, reason

        return False, ""

    def get_known_recipients_count(self, sender: str) -> int:
        return len(self.relationships[sender])

    def reset(self):
        self.relationships = defaultdict(set)


# ─────────────────────────────────────────────
# SHAP-STYLE EXPLAINABILITY MODULE
# ─────────────────────────────────────────────

def compute_risk_contributions(
    transaction: dict,
    fraud_prob: float,
    velocity_count: int,
    is_new_beneficiary: bool,
    device_risk: float,
    is_unusual_time: bool,
) -> list[dict]:
    """
    Compute human-readable risk factor contributions.
    Returns list of {factor, contribution, direction, description}.
    Inspired by SHAP additive explanations.
    """
    contributions = []
    amount = transaction.get("amount", 0)
    tx_type = transaction.get("type", "UNKNOWN")
    balance_drop = transaction.get("oldbalanceOrg", 0) - transaction.get("newbalanceOrig", 0)
    dest_balance_change = transaction.get("newbalanceDest", 0) - transaction.get("oldbalanceDest", 0)

    # ── Amount risk ──
    if amount > 1_000_000:
        amt_contrib = 45
        amt_desc = f"Very high amount (₹{amount:,.0f})"
    elif amount > 200_000:
        amt_contrib = 30
        amt_desc = f"High amount (₹{amount:,.0f})"
    elif amount > 50_000:
        amt_contrib = 15
        amt_desc = f"Moderate-high amount (₹{amount:,.0f})"
    else:
        amt_contrib = -5
        amt_desc = f"Normal amount (₹{amount:,.0f})"

    contributions.append({
        "factor": "Transaction Amount",
        "contribution": amt_contrib,
        "direction": "risk" if amt_contrib > 0 else "safe",
        "description": amt_desc,
        "icon": "💰"
    })

    # ── Transaction type risk ──
    type_risk = {"TRANSFER": 35, "CASH_OUT": 30, "CASH_IN": -10, "PAYMENT": -15, "DEBIT": 5}
    t_contrib = type_risk.get(tx_type, 0)
    contributions.append({
        "factor": "Transaction Type",
        "contribution": t_contrib,
        "direction": "risk" if t_contrib > 0 else "safe",
        "description": f"{tx_type} transactions carry {'elevated' if t_contrib > 0 else 'lower'} fraud risk",
        "icon": "📋"
    })

    # ── Balance zeroed out ──
    if balance_drop > 0 and transaction.get("newbalanceOrig", 1) == 0:
        contributions.append({
            "factor": "Account Drained",
            "contribution": 40,
            "direction": "risk",
            "description": "Account balance reduced to zero — classic fraud pattern",
            "icon": "🚨"
        })
    elif balance_drop > amount * 0.95:
        contributions.append({
            "factor": "Large Balance Drop",
            "contribution": 20,
            "direction": "risk",
            "description": f"Balance dropped by ₹{balance_drop:,.0f}",
            "icon": "📉"
        })

    # ── Destination balance unchanged (mule account) ──
    if dest_balance_change == 0 and amount > 10_000:
        contributions.append({
            "factor": "Destination Unchanged",
            "contribution": 25,
            "direction": "risk",
            "description": "Recipient balance didn't increase — funds may have moved on (mule account pattern)",
            "icon": "🔄"
        })

    # ── Velocity ──
    if velocity_count >= 5:
        v_contrib = min(10 * (velocity_count - 4), 40)
        contributions.append({
            "factor": "Transaction Velocity",
            "contribution": v_contrib,
            "direction": "risk",
            "description": f"{velocity_count} transactions in short window — rapid bursts indicate account takeover",
            "icon": "⚡"
        })
    elif velocity_count >= 3:
        contributions.append({
            "factor": "Transaction Velocity",
            "contribution": 10,
            "direction": "risk",
            "description": f"{velocity_count} transactions in recent window",
            "icon": "⚡"
        })

    # ── New beneficiary ──
    if is_new_beneficiary:
        contributions.append({
            "factor": "New Beneficiary",
            "contribution": 20,
            "direction": "risk",
            "description": "First-ever transfer to this recipient",
            "icon": "👤"
        })

    # ── Device risk ──
    if device_risk > 0.5:
        d_contrib = int(device_risk * 30)
        contributions.append({
            "factor": "Device Fingerprint",
            "contribution": d_contrib,
            "direction": "risk",
            "description": f"Unrecognized device (risk score: {device_risk:.0%})",
            "icon": "📱"
        })
    elif device_risk < 0.15:
        contributions.append({
            "factor": "Device Fingerprint",
            "contribution": -10,
            "direction": "safe",
            "description": "Known trusted device",
            "icon": "📱"
        })

    # ── Unusual hour ──
    if is_unusual_time:
        contributions.append({
            "factor": "Transaction Timing",
            "contribution": 20,
            "direction": "risk",
            "description": f"Late-night/early-morning transfer (hour {get_transaction_hour(transaction.get('step', 0)):02d}:00)",
            "icon": "🌙"
        })

    # Normalize so total positive contributions reflect fraud probability
    pos_total = sum(c["contribution"] for c in contributions if c["contribution"] > 0) or 1
    scale = min(fraud_prob * 100 / pos_total, 1.5)
    for c in contributions:
        c["scaled"] = round(c["contribution"] * scale, 1)

    return sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)


# ─────────────────────────────────────────────
# COMPOSITE RISK SCORER
# ─────────────────────────────────────────────

def compute_composite_risk(
    ml_prob: float,
    device_risk: float,
    velocity_flagged: bool,
    is_new_beneficiary: bool,
    is_unusual_time: bool,
    amount: float,
    tx_type: str,
) -> tuple[float, str]:
    """
    Combines ML model probability with rule-based signals.
    Returns (composite_score 0-1, risk_level string).
    """
    score = ml_prob

    # Additive risk signals (capped)
    if velocity_flagged:
        score += 0.15
    if is_new_beneficiary:
        score += 0.08
    if device_risk > 0.5:
        score += device_risk * 0.1
    if is_unusual_time:
        score += 0.07
    if tx_type in ("TRANSFER", "CASH_OUT") and amount > 200_000:
        score += 0.05

    score = min(score, 0.999)

    if score >= 0.85:
        level = "CRITICAL"
    elif score >= 0.65:
        level = "HIGH"
    elif score >= 0.40:
        level = "MEDIUM"
    elif score >= 0.20:
        level = "LOW"
    else:
        level = "SAFE"

    return score, level


RISK_COLORS = {
    "CRITICAL": "#FF2D55",
    "HIGH": "#FF6B35",
    "MEDIUM": "#FFB800",
    "LOW": "#00C7BE",
    "SAFE": "#34C759",
}
