from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from rag.core.config import Settings


def setup_tracing(settings: Settings) -> None:
    resource = Resource.create({"service.name": "rag-api", "service.version": "0.2.0"})
    provider = TracerProvider(resource=resource)

    # Console exporter for dev; replace with OTLP exporter for production
    processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)


def get_tracer(name: str = "rag"):
    return trace.get_tracer(name)
