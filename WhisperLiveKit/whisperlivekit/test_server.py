#!/usr/bin/env python3
"""
Test server for WhisperLiveKit with LibriSpeech-style finish command support
Based on basic_server.py but adds support for 'finish' command to flush buffers
"""
import asyncio
import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from whisperlivekit import AudioProcessor, TranscriptionEngine, parse_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def handle_websocket_results(websocket, results_generator, finish_event):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            response_dict = response.to_dict()
            logger.debug(f"Sending response: {response_dict}")
            await websocket.send_json(response_dict)

        # When results_generator finishes, signal ready to stop
        logger.info("Results generator finished.")
        finish_event.set()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results.")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    # Send hello message (LibriSpeech style)
    try:
        await websocket.send_json({
            "type": "hello",
            "message": "Connected to WhisperLiveKit Test Server"
        })
    except Exception as e:
        logger.warning(f"Failed to send hello message: {e}")

    # Send config message
    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    finish_event = asyncio.Event()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator, finish_event))

    try:
        while True:
            # Receive either bytes (audio) or text (JSON commands)
            try:
                message = await websocket.receive()

                # Check message type
                if "bytes" in message:
                    # Audio data
                    audio_data = message["bytes"]
                    await audio_processor.process_audio(audio_data)

                elif "text" in message:
                    # JSON command
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type", "")

                        if msg_type == "finish":
                            logger.info("Received finish command - flushing buffers")
                            # Signal audio processor to finish
                            await audio_processor.finish()

                            # Wait a bit for final results (timeout instead of waiting forever)
                            try:
                                await asyncio.wait_for(finish_event.wait(), timeout=5.0)
                                logger.info("Buffer flushed, all results sent")
                            except asyncio.TimeoutError:
                                logger.info("Timeout waiting for finish_event, proceeding to close")
                            break

                        elif msg_type == "stop":
                            logger.info("Received stop command")
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received: {message['text']}")

            except KeyError as e:
                # No bytes or text in message - connection likely closed
                if 'bytes' in str(e) or 'text' in str(e):
                    logger.warning(f"Client has closed the connection.")
                    break
                else:
                    logger.error(f"Unexpected KeyError: {e}", exc_info=True)
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


def main():
    """Entry point for the CLI command."""
    import uvicorn

    uvicorn_kwargs = {
        "app": "whisperlivekit.test_server:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if args.forwarded_allow_ips:
        uvicorn_kwargs = { **uvicorn_kwargs, "forwarded_allow_ips" : args.forwarded_allow_ips }

    logger.info(f"Starting WhisperLiveKit Test Server on {args.host}:{args.port}")
    logger.info(f"Backend policy: {args.backend_policy}")
    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
