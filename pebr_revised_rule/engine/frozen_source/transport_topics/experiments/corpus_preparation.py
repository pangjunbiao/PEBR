"""Corpus-preparation execution, with no topic fitting or performance selection."""
from ..data.sources import SOURCES, obtain
from ..representations.embeddings import FrozenEncoder
from ..data.preparation import prepare_collections
from ..data.preparation import frozen_json


def run(root,config,environment,logger,store):
    sources={name:obtain(root,value,logger,config['preparation']['download']) for name,value in SOURCES.items()}
    store.json('source_acquisition.json',sources)
    for name,source in sources.items():
        stable={k:v for k,v in source.items() if k!='downloaded'}
        frozen_json(root/'data/manifests/sources'/(name+'_'+source['sha256'][:16]+'.json'),stable)
    logger.info('Initializing the frozen ONNX encoder and verifying its compute provider.')
    encoder=FrozenEncoder(root,config,environment,logger,store.directory)
    store.json('encoder_audit.json',encoder.audit)
    summary=prepare_collections(root,encoder,config,logger,sources)
    store.json('corpus_preparation.json',summary)
    for condition in summary['conditions']:
        logger.info('COLLECTION COMPLETE | %s | posts=%d vocabulary=%d families=%d | %s',
                    condition['name'],condition['posts'],condition['vocabulary'],condition['families'],condition['path'])
    logger.info('Preparation scope: %s; registered full preparation complete=%s; topic models fitted=False.',
                summary['scope'],summary['complete_registered_preparation'])
    return summary
