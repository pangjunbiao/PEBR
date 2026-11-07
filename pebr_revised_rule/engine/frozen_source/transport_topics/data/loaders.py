"""Text records and evaluation annotations leave these loaders separately."""
from dataclasses import dataclass
from pathlib import Path
import csv
import io

SUBSTANTIVE = ('Congestion','Customer_interface','Delay','Personnel_attitude',
 'Physical_accessibility','Pricing','Refusal_cancellation_refund','Safety',
 'Unavailability_inadequacy','Vehicle_utility_condition')


@dataclass(frozen=True)
class Post:
    source_row: int
    source_id: str
    original_text: str


def load_primary(path: Path):
    posts, annotations = [], {}
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        expected = {'text', *SUBSTANTIVE, 'Miscellaneous', 'Not_applicable'}
        if set(reader.fieldnames or []) != expected:
            raise ValueError('Unexpected Indian grievance CSV schema')
        for row_index, row in enumerate(reader):
            if any(row[k] not in ('0','1') for k in expected-{'text'}):
                raise ValueError('Nonbinary primary annotation at source row %d' % row_index)
            posts.append(Post(row_index, 'indian_transport:%d' % row_index, row['text']))
            annotations[row_index] = [k for k in SUBSTANTIVE if row[k]=='1']
    if not posts: raise ValueError('Empty primary CSV')
    return posts, annotations


def load_external(path: Path):
    # The author's three-class training file has NO header. Retain its first row.
    posts, total, filtered = [], 0, 0
    with path.open(encoding='utf-8-sig', newline='') as source:
        for row_index, row in enumerate(csv.reader(source)):
            total += 1
            if len(row)!=3 or row[0] not in ('0','1','2') or not row[2].strip():
                raise ValueError('Malformed headerless external CSV at source row %d' % row_index)
            if row[0]=='0':
                filtered += 1; continue
            identifier = row[1][1:] if row[1].startswith('s') else row[1]
            posts.append(Post(row_index, 'dabiri_traffic:'+identifier, row[2]))
    return posts, {'raw_rows':total, 'transport_rows':len(posts), 'nontraffic_filtered':filtered,
                   'label_use':'One-time transport population filter; class values do not enter fitting'}
