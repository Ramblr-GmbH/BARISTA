"""Task implementations for VLM benchmarks."""

from barista.vlm_benchmarks.tasks.activity_mcq import ActivityMcqTask
from barista.vlm_benchmarks.tasks.base import BenchmarkAssetPreparer, BenchmarkExampleBuilder, BenchmarkTask
from barista.vlm_benchmarks.tasks.grounding import GroundingTask
from barista.vlm_benchmarks.tasks.hand_object import HandObjectTask
from barista.vlm_benchmarks.tasks.referring import ReferringBuildTask, ReferringRunTask
from barista.vlm_benchmarks.tasks.relation_extraction import RelationExtractionTask
from barista.vlm_benchmarks.tasks.visual_qa import VisualQABuildTask, VisualQAPrepareTask, VisualQARunTask
from barista.vlm_benchmarks.types import DatasetBuildConfig, DatasetPrepareConfig, RunConfig


def create_prepare_task(config: DatasetPrepareConfig) -> BenchmarkAssetPreparer:
    if config.task == 'visual_qa':
        return VisualQAPrepareTask(
            dataset_root=config.dataset_root,
            task_params=config.task_params,
            config=config,
        )
    raise ValueError(f'Task does not support separate asset preparation: {config.task}')


def create_build_task(config: DatasetBuildConfig) -> BenchmarkExampleBuilder:
    if config.task == 'activity_mcq':
        return ActivityMcqTask(dataset_root=config.dataset_root, task_params=config.task_params)
    if config.task == 'grounding':
        return GroundingTask(dataset_root=config.dataset_root, task_params=config.task_params)
    if config.task == 'hand_object':
        return HandObjectTask(dataset_root=config.dataset_root, task_params=config.task_params)
    if config.task == 'visual_qa':
        return VisualQABuildTask(
            dataset_root=config.dataset_root,
            task_params=config.task_params,
        )
    if config.task == 'referring':
        return ReferringBuildTask(
            dataset_root=config.dataset_root,
            task_params=config.task_params,
        )
    if config.task == 'relation_extraction':
        return RelationExtractionTask(dataset_root=config.dataset_root, task_params=config.task_params)
    raise ValueError(f'Unsupported task: {config.task}')


def create_run_task(config: RunConfig) -> BenchmarkTask:
    if config.task == 'activity_mcq':
        return ActivityMcqTask(dataset_root=config.dataset_root, task_params=config.task_params)
    if config.task == 'grounding':
        return GroundingTask(dataset_root=config.dataset_root, task_params=config.task_params)
    if config.task == 'hand_object':
        return HandObjectTask(dataset_root=config.dataset_root, task_params=config.task_params)
    if config.task == 'visual_qa':
        return VisualQARunTask(
            dataset_root=config.dataset_root,
            task_params=config.task_params,
            run_config=config,
        )
    if config.task == 'referring':
        return ReferringRunTask(
            dataset_root=config.dataset_root,
            task_params=config.task_params,
            run_config=config,
        )
    if config.task == 'relation_extraction':
        return RelationExtractionTask(dataset_root=config.dataset_root, task_params=config.task_params)
    raise ValueError(f'Unsupported task: {config.task}')


__all__ = [
    'ActivityMcqTask',
    'BenchmarkAssetPreparer',
    'BenchmarkExampleBuilder',
    'BenchmarkTask',
    'GroundingTask',
    'HandObjectTask',
    'ReferringBuildTask',
    'ReferringRunTask',
    'RelationExtractionTask',
    'VisualQABuildTask',
    'VisualQAPrepareTask',
    'VisualQARunTask',
    'create_build_task',
    'create_prepare_task',
    'create_run_task',
]
