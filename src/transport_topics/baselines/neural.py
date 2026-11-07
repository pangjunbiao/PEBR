"""Common training adapter around unchanged pinned GloCOM/kNNTM networks."""
from pathlib import Path
import csv
import io
import json
import time
import numpy as np

from ..runtime.artifacts import file_hash, write_json, atomic_write
from ..representations.word_embeddings import load_word_vectors
from .sources import prepare_sources, load_native
from .inputs import global_context, transport_distances
from .base import ModelOutput


def training_batches(size, batch_size, permutation):
    if size < 2:
        raise ValueError('Published batch-normalized networks require at least two documents')
    chunks = [permutation[i:i+batch_size] for i in range(0, size, batch_size)]
    if len(chunks[-1]) == 1:
        if len(chunks) == 1:
            raise ValueError('A singleton batch cannot train a batch-normalized model')
        import torch
        chunks[-2] = torch.cat((chunks[-2], chunks[-1])); chunks.pop()
    return chunks


def _save_checkpoint(directory, model, optimizer, completed_epoch, losses, identity, device):
    import torch
    state = {'identity': identity, 'epoch': completed_epoch, 'model': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'losses': losses,
             'cpu_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/('epoch_%04d.pt' % completed_epoch)
    temporary = path.with_suffix('.partial.pt')
    torch.save(state, temporary); temporary.replace(path)
    write_json(directory/'latest.json', {'identity': identity, 'epoch': completed_epoch,
                                         'filename':path.name,'sha256':file_hash(path)}, overwrite=True)
    for old in sorted(directory.glob('epoch_*.pt'))[:-2]:
        old.unlink()
    return path


def fit(root, collection, method, topics, seed, settings, integration, device_name,
        checkpoint_directory, model_identity, logger, epochs=None):
    import torch
    if method not in ('knntm','glocom'):
        raise ValueError('Unknown published neural model')
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source_directory, source_audit = prepare_sources(root, logger, integration['download_sources'])
    native = load_native(source_directory, method)
    counts = collection.counts.toarray().astype(np.float32)
    vocabulary = np.load(collection.directory/'vocabulary.npy', allow_pickle=False)
    embeddings = np.load(collection.directory/'embeddings.npy', allow_pickle=False)
    dataset = torch.as_tensor(counts, device=device)
    input_audit = {}
    if method=='glocom':
        vectors, word_audit = load_word_vectors(root, vocabulary, 200)
        context, groups = global_context(collection.counts, embeddings, settings['clusters'], settings['cluster_seed'])
        input_data = torch.as_tensor(np.concatenate((counts,context),axis=1), device=device)
        model = native(counts.shape[1], num_topics=topics, en_units=settings['hidden_units'],
                       pretrained_WE=vectors, aug_coef=settings['augmentation'],
                       prior_var=settings['prior_variance'], weight_loss_ECR=settings['ecr_weight'],
                       dropout=settings['dropout'], beta_temp=settings['beta_temperature'],
                       sinkhorn_alpha=settings['sinkhorn_alpha'], sinkhorn_max_iter=settings['sinkhorn_max_iterations'])
        input_audit = {'word_vectors': word_audit, 'clusters_requested': settings['clusters'],
                       'clusters_actual': len(np.unique(groups)), 'clustering_seed': settings['cluster_seed'],
                       'context': 'sum of original BOW rows in each fixed MiniLM KMeans group',
                       'clustering_n_init': 10}
    else:
        vectors, word_audit = load_word_vectors(root, vocabulary, 300)
        distance_directory, distance_audit = transport_distances(root, collection, vectors, word_audit, logger)
        input_data = dataset
        model = native(counts.shape[1], len(counts), dataset,
                       str(distance_directory/'M_cos.npz'), str(distance_directory/'M_coo.npz'),
                       num_topics=topics, en_units=settings['hidden_units'], alpha=settings['alpha'],
                       num_k=min(settings['neighbors'],len(counts)-1), eta=settings['eta'], rho=settings['rho'])
        input_audit = {'word_vectors': word_audit, 'transport': distance_audit,
                       'neighbors_actual': min(settings['neighbors'],len(counts)-1),
                       'neural_implementation': 'released GloCOM reproduction, including its topic-distance calculation'}
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    target_epochs = settings['epochs'] if epochs is None else int(epochs)
    if method=='knntm' and target_epochs <= settings['warmup_epochs']:
        raise ValueError('A kNNTM run must include augmented epochs beyond its warmup')
    checkpoint_index = checkpoint_directory/'latest.json'
    checkpoint = None
    losses, first_epoch = [], 1
    if checkpoint_index.exists():
        receipt = json.loads(checkpoint_index.read_text(encoding='utf-8'))
        filename = receipt['filename']
        if Path(filename).name != filename:
            raise ValueError('Invalid neural checkpoint filename')
        checkpoint = checkpoint_directory/filename
        if receipt['identity'] != model_identity or receipt['sha256'] != file_hash(checkpoint):
            raise ValueError('Neural checkpoint identity/checksum mismatch')
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state['identity'] != model_identity:
            raise ValueError('Neural checkpoint belongs to another model condition')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        losses, first_epoch = state['losses'], int(state['epoch'])+1
        logger.info('Resuming %s at epoch %d/%d.',method,first_epoch,target_epochs)
    began = time.perf_counter(); steps = 0
    for epoch in range(first_epoch,target_epochs+1):
        model.train()
        permutation = torch.randperm(len(counts))
        total = 0.0
        for ids in training_batches(len(counts),settings['batch_size'],permutation):
            ids = ids.to(device)
            batch = input_data[ids]
            result = model(ids,batch,epoch>settings['warmup_epochs']) if method=='knntm' else model(batch)
            loss = result['loss']
            if not torch.isfinite(loss).item():
                raise ArithmeticError('Nonfinite published model training loss: '+method)
            optimizer.zero_grad();loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise ArithmeticError('Nonfinite published model gradient: '+method)
            optimizer.step();steps+=1
            total += float(loss.detach().cpu())*len(ids)
        losses.append({'epoch':epoch,'loss':total/len(counts)})
        if epoch==1 or epoch%5==0 or epoch==target_epochs:
            logger.info('Neural training | %s K=%d seed=%d | epoch %d/%d loss %.5f | %s.',
                        method,topics,seed,epoch,target_epochs,total/len(counts),device)
        if epoch%integration['checkpoint_epochs']==0 or epoch==target_epochs:
            checkpoint = _save_checkpoint(checkpoint_directory,model,optimizer,epoch,losses,model_identity,device)
    if device.type=='cuda':torch.cuda.synchronize(device)
    model.eval()
    with torch.no_grad():
        theta = torch.cat([model.get_theta(input_data[start:start+settings['batch_size']])
                           for start in range(0,len(counts),settings['batch_size'])]).cpu().numpy()
        beta = model.get_beta().cpu().numpy()
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=['epoch','loss']);writer.writeheader();writer.writerows(losses)
    atomic_write(checkpoint_directory/'loss_history.csv',stream.getvalue().encode('utf-8'),overwrite=True)
    audit = {'implementation': source_audit, 'input_preparation': input_audit,
             'epochs':target_epochs,'completed_epochs':target_epochs,
             'augmented_epochs':max(0,target_epochs-settings['warmup_epochs']) if method=='knntm' else target_epochs,
             'training_device':str(device),'cuda_execution_verified':device.type=='cuda' and (steps>0 or bool(losses)),
             'termination':'fixed published epoch budget','optimizer':'Adam',
             'settings':settings,'seconds_this_invocation':time.perf_counter()-began,
             'checkpoint':checkpoint.relative_to(root).as_posix(),
             'checkpoint_sha256':file_hash(checkpoint),
             'topic_score_kind':'signed decoder logits' if method=='knntm' else 'native embedding-decoder scores',
             'retrieval_representation':'native get_theta on original document and its prescribed native context',
             'singleton_batch_policy':'merge final singleton into previous batch; never discard a document',
             'namespace_import_rewrite_only':True,'annotations_used':False}
    return ModelOutput(beta,theta,theta.copy(),audit)
