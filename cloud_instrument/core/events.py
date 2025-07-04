"""Event management system for Cloud Instrument."""

import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the system."""
    OSC_MESSAGE = "osc_message"
    AUDIO_STATE_CHANGE = "audio_state_change"
    MIDI_MESSAGE = "midi_message"
    SOUND_LOADED = "sound_loaded"
    SOUND_PLAY_START = "sound_play_start"
    SOUND_PLAY_STOP = "sound_play_stop"
    MIR_STATE_UPDATE = "mir_state_update"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_SHUTDOWN = "system_shutdown"


@dataclass
class Event:
    """Event data structure."""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventManager:
    """Manages event subscriptions and dispatching."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._running = False
        
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe to an event type.
        
        Args:
            event_type: The type of event to subscribe to
            callback: Function to call when event occurs
        """
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value} events")
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from an event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            callback: The callback function to remove
        """
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from {event_type.value} events")
    
    def emit(self, event_type: EventType, data: Dict[str, Any], source: Optional[str] = None) -> None:
        """Emit an event to all subscribers.
        
        Args:
            event_type: The type of event
            data: Event data
            source: Optional source identifier
        """
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source
        )
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify subscribers
        subscribers = self._subscribers[event_type]
        logger.debug(f"Emitting {event_type.value} to {len(subscribers)} subscribers")
        
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback for {event_type.value}: {e}")
    
    async def emit_async(self, event_type: EventType, data: Dict[str, Any], source: Optional[str] = None) -> None:
        """Emit an event asynchronously.
        
        Args:
            event_type: The type of event
            data: Event data  
            source: Optional source identifier
        """
        # Run synchronous emit in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.emit, event_type, data, source)
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get recent event history.
        
        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.debug("Event history cleared")
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type.
        
        Args:
            event_type: The event type
            
        Returns:
            Number of subscribers
        """
        return len(self._subscribers[event_type])
    
    def get_all_subscribers(self) -> Dict[EventType, int]:
        """Get subscriber counts for all event types.
        
        Returns:
            Dictionary mapping event types to subscriber counts
        """
        return {event_type: len(callbacks) for event_type, callbacks in self._subscribers.items()}


# Global event manager instance
event_manager = EventManager()