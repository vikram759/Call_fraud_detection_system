# src/synth_data.py
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import uuid

def generate_msisdn(n):
    return [f"9{random.randint(10**8,10**9-1)}" for _ in range(n)]

def generate_imei(n):
    return [str(350000000000000 + i) for i in range(n)]

def pick_time_series(start, minutes, n_events):
    times = []
    for _ in range(n_events):
        offset = random.randint(0, minutes*60)
        times.append(start + timedelta(seconds=offset))
    times.sort()
    return times

def create_synthetic_log(n_users=500, duration_days=1, avg_events_per_user=50, fraud_fraction=0.02):
    rows = []
    start = datetime.now(timezone.utc)  # ✅ modern UTC-aware datetime
    msisdns = generate_msisdn(n_users)
    imeis = generate_imei(n_users)
    gateway_ips = [f"10.0.{i//256}.{i%256}" for i in range(100)]

    for i, ms in enumerate(msisdns):
        imei = imeis[i]  
        n_events = max(1, int(np.random.poisson(avg_events_per_user)))
        times = pick_time_series(start, duration_days * 24 * 60, n_events)
        for t in times:
            evt = random.choices(['CALL', 'SMS', 'DATA'], weights=[0.6, 0.3, 0.1])[0]
            if evt == 'CALL':
                if random.random() < 0.95:
                    callee = random.choice(msisdns)
                else:
                    callee = f"44{random.randint(10**8,10**9-1)}"
                duration = random.expovariate(1/60) if random.random() > 0.2 else random.expovariate(1/5)
            elif evt == 'SMS':
                callee = random.choice(msisdns)
                duration = 0
            else:
                callee = ''
                duration = random.randint(1000, 100000)

            gateway = random.choice(gateway_ips)
            sms_text = ''
            domain = ''
            if evt == 'SMS' and random.random() < 0.05:
                sms_text = f"Click http://{uuid.uuid4().hex[:6]}.biz"
                domain = sms_text.split("//")[1]
            if evt == 'DATA' and random.random() < 0.02:
                domain = f"{uuid.uuid4().hex[:8]}.xyz"

            rows.append({
                'timestamp': t,
                'caller': ms,
                'callee': callee,
                'imei': imei,
                'gateway': gateway,
                'event': evt,
                'duration': duration,
                'sms_text': sms_text,
                'domain': domain,
                'label': 'normal'
            })

    df = pd.DataFrame(rows)
    df = inject_frauds(df, msisdns)
    return df

def inject_frauds(df, msisdns):
    """
    Inject synthetic fraud events (CALL and SMS) into the dataframe.

    Parameters:
    - df: Original dataframe containing call/SMS logs
    - msisdns: List of user phone numbers

    Returns:
    - df: DataFrame with additional fraud events
    """
    n = len(msisdns)
    fraud_users = random.sample(msisdns, max(1, int(1 * n)))
    fraud_rows = []  # collect new rows

    # Ensure df['timestamp'] is tz-aware UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    start_min = df['timestamp'].min()
    start_max = df['timestamp'].max()
    total_seconds = (start_max - start_min).total_seconds()

    for fu in fraud_users:
        # Get IMEI and random gateway for this user
        imei_fu = df[df['caller'] == fu]['imei'].iloc[0]
        gateway_sample = df['gateway'].sample(1).iloc[0]

        # Random start timestamp within min and max
        random_offset_seconds = np.random.rand() * total_seconds
        random_start = start_min + pd.to_timedelta(random_offset_seconds, unit='s')

        # Generate CALL events
        call_times = pd.date_range(start=random_start, periods=20, freq='min')  # 20 calls, 1 min apart
        for t in call_times:
            fraud_rows.append({
                'timestamp': t,
                'caller': fu,
                'callee': f"44{random.randint(10**8, 10**9-1)}",
                'imei': imei_fu,
                'gateway': gateway_sample,
                'event': 'CALL',
                'duration': random.uniform(0.5, 2),  # minutes
                'sms_text': '',
                'domain': '',
                'label': 'wangiri'
            })

        # Generate SMS spam events
        sms_times = pd.date_range(start=start_min, periods=30, freq='30s')  # 30 SMS, 30 seconds apart
        for t in sms_times:
            fraud_rows.append({
                'timestamp': t,
                'caller': fu,
                'callee': random.choice(msisdns),
                'imei': imei_fu,
                'gateway': gateway_sample,
                'event': 'SMS',
                'duration': 0,
                'sms_text': f"Buy now {uuid.uuid4().hex[:4]} http://{uuid.uuid4().hex[:5]}.biz",
                'domain': f"{uuid.uuid4().hex[:5]}.biz",
                'label': 'spam_sms'
            })

    # Append new fraud rows
    if fraud_rows:
        df = pd.concat([df, pd.DataFrame(fraud_rows)], ignore_index=True)

    # Ensure timestamps remain UTC and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = create_synthetic_log(n_users=200, duration_days=1, avg_events_per_user=60)
    df.to_csv("data/raw_logs.csv", index=False)
    print("✅ Saved data/raw_logs.csv with", len(df), "rows")
