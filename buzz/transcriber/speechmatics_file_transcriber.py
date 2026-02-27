import asyncio
import logging
import time
from typing import Optional, List

from PyQt6.QtCore import QObject
from speechmatics.batch._models import Transcript

from buzz.settings.settings import Settings
from buzz.transcriber.file_transcriber import FileTranscriber
from buzz.transcriber.transcriber import FileTranscriptionTask, Segment, Stopped


class SpeechmaticsFileTranscriber(FileTranscriber):
    """Batch transcription via the Speechmatics cloud API (speechmatics-batch package)."""

    def __init__(self, task: FileTranscriptionTask, parent: Optional["QObject"] = None):
        super().__init__(task=task, parent=parent)
        self._stopped = False
        settings = Settings()
        self.custom_speechmatics_url = settings.value(
            key=Settings.Key.CUSTOM_SPEECHMATICS_URL, default_value=""
        )
        logging.debug("Will use speechmatics API on %s", self.custom_speechmatics_url)

    def transcribe(self) -> List[Segment]:
        api_key = self.transcription_task.transcription_options.speechmatics_access_token
        if not api_key:
            raise Exception(
                "Speechmatics API key is not set. Please enter your API key in the settings."
            )

        language = self.transcription_task.transcription_options.language or "auto"
        file_path = self.transcription_task.file_path
        identify_speaker = self.transcription_task.transcription_options.identify_speaker

        logging.debug(
            "Starting Speechmatics batch transcription: file=%s, language=%s",
            file_path,
            language,
        )

        self.progress.emit((0, 100))

        return asyncio.run(self._run_transcription(api_key, language, file_path, identify_speaker))

    async def _run_transcription(
        self, api_key: str, language: str, file_path: str, identify_speaker: bool
    ) -> List[Segment]:
        from speechmatics.batch import (
            AsyncClient,
            JobConfig,
            JobType,
            JobStatus,
            TranscriptionConfig,
            FormatType,
        )

        async with AsyncClient(
            api_key=api_key,
            url=self.custom_speechmatics_url if self.custom_speechmatics_url else None,
        ) as client:
            config = JobConfig(
                type=JobType.TRANSCRIPTION,
                transcription_config=TranscriptionConfig(language=language),
            )

            logging.debug("Submitting Speechmatics job for file: %s", file_path)
            job = await client.submit_job(file_path, config=config)
            logging.debug("Speechmatics job submitted: id=%s", job.id)

            self.progress.emit((10, 100))

            # Poll until complete or stopped
            poll_interval = 3.0
            while True:
                if self._stopped:
                    logging.debug("Speechmatics transcription stopped by user")
                    raise Stopped()

                job_info = await client.get_job_info(job.id)
                status = job_info.status
                logging.debug("Speechmatics job %s status: %s", job.id, status)

                if status == JobStatus.DONE:
                    break
                elif status in (JobStatus.REJECTED, JobStatus.DELETED):
                    raise Exception(
                        f"Speechmatics job failed with status: {status.value}"
                    )

                # Emit a rough progress update while waiting
                self.progress.emit((20, 100))
                await asyncio.sleep(poll_interval)

            # Retrieve JSON transcript
            logging.debug("Retrieving Speechmatics transcript for job: %s", job.id)
            self.progress.emit((90, 100))
            result = await client.get_transcript(job.id, format_type=FormatType.JSON)

            segments = self._parse_segments(result, identify_speaker)
            self.progress.emit((100, 100))
            logging.debug(
                "Speechmatics transcription complete: %d segments", len(segments)
            )
            return segments

    def _parse_segments(self, transcript: Transcript, identify_speaker: bool) -> List[Segment]:
        """Parse Speechmatics JSON result into Segment objects.

        The JSON-v2 result has a `results` list where each item has:
          - start_time (float, seconds)
          - end_time (float, seconds)
          - alternatives: list of dicts with 'content' key
          - type: 'word' | 'punctuation'
        We group consecutive words/punctuation into sentence-level segments
        by collapsing them all into a single segment per result item.
        """
        if not transcript.results:
            return []

        # Get language pack info for word delimiter
        word_delimiter = " "  # Default
        if transcript.metadata and transcript.metadata.language_pack_info and "word_delimiter" in transcript.metadata.language_pack_info:
            word_delimiter = transcript.metadata.language_pack_info["word_delimiter"]

        # Group results by speaker and process
        segments = []
        current_speaker = None
        current_start = None
        current_end = None
        current_group: list[str] = []
        prev_eos = "word"

        for result in transcript.results:
            if not result.alternatives:
                continue

            alternative = result.alternatives[0]
            content = alternative.content
            speaker = alternative.speaker

            # Handle speaker changes, end of sentence or if segment is too long
            if speaker != current_speaker or prev_eos or (current_end - current_start > 60):
                # Process accumulated group for previous speaker
                if current_group:
                    text = transcript._join_content_items(current_group, word_delimiter)
                    if current_speaker:
                        text = f"{current_speaker}: {text}" if identify_speaker else text
                        segments.append(
                            Segment(
                                start=int(current_start * 1000),
                                end=int(current_end * 1000),
                                text=text,
                            )
                        )
                    else:
                        segments.append(Segment(
                            start=int(current_start * 1000),
                            end=int(current_end * 1000),
                            text=text,
                        ))
                    current_group = []
                    current_start = None

                current_speaker = speaker

            # Add content to current group
            if content:
                current_group.append(content)
                prev_eos = result.type == "punctuation" and content in ".?!"

                if current_start is None:
                    current_start = result.start_time
                current_end = result.end_time

        # Process final group
        if current_group:
            text = transcript._join_content_items(current_group, word_delimiter)
            if current_speaker:
                text = f"{current_speaker}: {text}" if identify_speaker else text
                segments.append(
                    Segment(
                        start=int(current_start * 1000),
                        end=int(current_end * 1000),
                        text=text,
                    )
                )
            else:
                segments.append(Segment(
                    start=int(current_start * 1000),
                    end=int(current_end * 1000),
                    text=text,
                ))

        return segments

    def stop(self):
        self._stopped = True
