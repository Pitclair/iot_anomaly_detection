import pandas as pd
from lm_idnet.processing.aggregation import aggregate_packet_traces


def test_aggregate_simple():
    data = {
        'timestamp': ['2020-01-01 00:00:00', '2020-01-01 00:01:00', '2020-01-01 00:09:59', '2020-01-01 00:10:00'],
        'protocol': ['TCP', 'UDP', 'TCP', 'ARP']
    }
    df = pd.DataFrame(data)
    out = aggregate_packet_traces(df, window_minutes=10, categories=['TCP','UDP','SSSDP','ARP'])
    # Expect two windows: first contains 3 events (TCP,UDP,TCP), second contains 1 (ARP)
    assert out.shape[0] == 2
    # note: some environments may compress windows differently; check sums
    assert out['TCP'].sum() == 2
    assert out['ARP'].sum() == 1
