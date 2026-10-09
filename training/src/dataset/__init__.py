"""WebDataset decoding and local shard resolution."""

import json
import os
from itertools import islice
from typing import Any, Optional

import datasets
import pyarrow as pa
from datasets.data_files import (
    _get_origin_metadata,
    DataFilesList as DataFilesListBase,
    DownloadConfig,
    SingleOriginMetadata,
)
from datasets.features.features import cast_to_python_objects
from datasets.packaged_modules.webdataset.webdataset import WebDataset as WebDatasetBase


class WebDataset(WebDatasetBase):
    def _split_generators(self, dl_manager: Any) -> list[datasets.SplitGenerator]:
        """Handle string/list/dict data files and infer features from one sample stream."""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download(self.config.data_files)
        splits: list[datasets.SplitGenerator] = []
        first_tar_path: str | None = None
        first_tar_iterator: Any | None = None
        for split_name, tar_paths in data_files.items():
            if isinstance(tar_paths, str):
                tar_paths = [tar_paths]
            tar_iterators = [dl_manager.iter_archive(tar_path) for tar_path in tar_paths]
            if first_tar_path is None:
                first_tar_path = tar_paths[0]
                first_tar_iterator = tar_iterators[0]
            splits.append(
                datasets.SplitGenerator(
                    name=split_name, gen_kwargs={"tar_paths": tar_paths, "tar_iterators": tar_iterators}
                )
            )
        if not self.info.features:
            if first_tar_path is None or first_tar_iterator is None:
                raise ValueError("No TAR files were resolved from data_files.")
            pipeline = self._get_pipeline_from_tar(first_tar_path, first_tar_iterator)
            first_examples = list(islice(pipeline, self.NUM_EXAMPLES_FOR_FEATURES_INFERENCE))
            if not first_examples:
                raise ValueError("Unable to infer features: no examples found in TAR pipeline.")
            if any(example.keys() != first_examples[0].keys() for example in first_examples):
                raise ValueError(
                    "The TAR archives of the dataset should be in WebDataset format, "
                    "but the files in the archive don't share the same prefix or the same types."
                )
            pa_tables = [
                pa.Table.from_pylist(cast_to_python_objects([example], only_1d_for_numpy=True))
                for example in first_examples
            ]
            inferred_arrow_schema = pa.concat_tables(pa_tables, promote_options="default").schema
            features = datasets.Features.from_arrow_schema(inferred_arrow_schema)

            for field_name in first_examples[0]:
                extension = field_name.rsplit(".", 1)[-1]
                if extension in self.IMAGE_EXTENSIONS:
                    features[field_name] = datasets.Image()
            for field_name in first_examples[0]:
                extension = field_name.rsplit(".", 1)[-1]
                if extension in self.AUDIO_EXTENSIONS:
                    features[field_name] = datasets.Audio()
            for field_name in first_examples[0]:
                extension = field_name.rsplit(".", 1)[-1]
                if extension in self.VIDEO_EXTENSIONS:
                    features[field_name] = datasets.Video()

            for field_name in first_examples[0]:
                extension = field_name.rsplit(".", 1)[-1]
                if extension in ["json"]:
                    features[field_name] = datasets.Value("string")

            self.info.features = features

        return splits


def json_loads(data: str | bytes | bytearray) -> str:
    """Parse JSON payload and normalize it back to a string value."""
    data_obj = json.loads(data)
    json_str = json.dumps(data_obj)
    return json_str


DECODERS = {
    "json": json_loads,
}
WebDataset.DECODERS = DECODERS


def _get_single_origin_metadata(
    data_file: str,
    download_config: Optional[DownloadConfig] = None,
) -> SingleOriginMetadata:
    _ = data_file, download_config
    return ()


class DataFilesList(DataFilesListBase):
    @classmethod
    def from_patterns(
        cls,
        patterns: list[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        _ = base_path, allowed_extensions
        data_files = patterns
        if not data_files:
            raise ValueError("`patterns` must contain at least one path.")
        if any("://" in str(path) for path in data_files):
            raise ValueError("Dataset manifests must list local TAR files, not remote URLs.")
        missing_files = [path for path in data_files if not os.path.exists(path)]
        if missing_files:
            preview = ", ".join(missing_files[:3])
            if len(missing_files) > 3:
                preview += f" ... (+{len(missing_files) - 3} more)"
            raise FileNotFoundError(f"Missing data file(s): {preview}")
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return cls(data_files, origin_metadata)


datasets.data_files._get_single_origin_metadata = _get_single_origin_metadata
datasets.data_files.DataFilesList = DataFilesList
datasets.packaged_modules.webdataset.WebDataset = WebDataset
