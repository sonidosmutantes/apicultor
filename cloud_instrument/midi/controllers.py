"""MIDI controller implementations."""

import logging
from typing import Optional, Callable, Dict, Any, List
from abc import ABC, abstractmethod

from midi.messages import MIDIMessage, MIDIMessageType
from core.events import EventManager, EventType

logger = logging.getLogger(__name__)


class MIDIController(ABC):
    """Abstract base class for MIDI controllers."""
    
    def __init__(self, name: str, event_manager: EventManager):
        self.name = name
        self.event_manager = event_manager
        self._message_handler: Optional[Callable[[MIDIMessage], None]] = None
    
    @abstractmethod
    def process_message(self, message: MIDIMessage) -> None:
        """Process incoming MIDI message.
        
        Args:
            message: MIDI message to process
        """
        pass
    
    def set_message_handler(self, handler: Callable[[MIDIMessage], None]) -> None:
        """Set custom message handler.
        
        Args:
            handler: Function to handle MIDI messages
        """
        self._message_handler = handler
    
    def handle_message(self, message: MIDIMessage) -> None:
        """Handle incoming MIDI message.
        
        Args:
            message: MIDI message to handle
        """
        if self._message_handler:
            self._message_handler(message)
        else:
            self.process_message(message)


class GenericMIDIController(MIDIController):
    """Generic MIDI controller with configurable mappings."""
    
    def __init__(self, name: str, event_manager: EventManager):
        super().__init__(name, event_manager)
        self._cc_mappings: Dict[int, str] = {}
        self._note_mappings: Dict[int, str] = {}
    
    def map_cc_to_parameter(self, cc_number: int, parameter_name: str) -> None:
        """Map CC number to parameter name.
        
        Args:
            cc_number: MIDI CC number (0-127)
            parameter_name: Parameter name to map to
        """
        self._cc_mappings[cc_number] = parameter_name
        logger.debug(f"Mapped CC {cc_number} to parameter '{parameter_name}'")
    
    def map_note_to_action(self, note_number: int, action_name: str) -> None:
        """Map note number to action name.
        
        Args:
            note_number: MIDI note number (0-127)
            action_name: Action name to map to
        """
        self._note_mappings[note_number] = action_name
        logger.debug(f"Mapped note {note_number} to action '{action_name}'")
    
    def process_message(self, message: MIDIMessage) -> None:
        """Process MIDI message with configured mappings."""
        if message.message_type == MIDIMessageType.CONTROL_CHANGE:
            self._handle_control_change(message)
        elif message.message_type in [MIDIMessageType.NOTE_ON, MIDIMessageType.NOTE_OFF]:
            self._handle_note_message(message)
    
    def _handle_control_change(self, message: MIDIMessage) -> None:
        """Handle control change message."""
        cc_number = message.controller_number
        cc_value = message.normalized_controller_value
        
        if cc_number in self._cc_mappings:
            parameter_name = self._cc_mappings[cc_number]
            
            # Emit audio state change event
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": parameter_name, "value": cc_value},
                source=f"MIDI:{self.name}"
            )
            
            logger.debug(f"CC {cc_number} -> {parameter_name} = {cc_value}")
        else:
            logger.debug(f"Unmapped CC {cc_number} = {message.controller_value}")
    
    def _handle_note_message(self, message: MIDIMessage) -> None:
        """Handle note on/off message."""
        note_number = message.note_number
        velocity = message.normalized_velocity
        
        if note_number in self._note_mappings:
            action_name = self._note_mappings[note_number]
            
            if message.message_type == MIDIMessageType.NOTE_ON and velocity > 0:
                # Emit action event
                self.event_manager.emit(
                    EventType.MIDI_MESSAGE,
                    {"action": action_name, "velocity": velocity, "note": note_number},
                    source=f"MIDI:{self.name}"
                )
                
                logger.debug(f"Note {note_number} ON -> action '{action_name}' vel={velocity}")
            elif message.message_type == MIDIMessageType.NOTE_OFF or velocity == 0:
                # Note off event
                self.event_manager.emit(
                    EventType.MIDI_MESSAGE,
                    {"action": f"{action_name}_off", "note": note_number},
                    source=f"MIDI:{self.name}"
                )
                
                logger.debug(f"Note {note_number} OFF -> action '{action_name}_off'")
        else:
            logger.debug(f"Unmapped note {note_number} {message.message_type.value}")


class YaeltexController(MIDIController):
    """Yaeltex controller with predefined mappings."""
    
    def __init__(self, event_manager: EventManager):
        super().__init__("Yaeltex", event_manager)
        self._setup_default_mappings()
    
    def _setup_default_mappings(self) -> None:
        """Set up default Yaeltex controller mappings."""
        # Volume controls (faders)
        self._volume_ccs = [0, 1, 2, 3, 4, 5, 6, 7]
        
        # Effect controls (knobs)
        self._pan_cc = 16
        self._reverb_room_cc = 17
        self._reverb_damping_cc = 18
        self._delay_time_cc = 19
        self._delay_feedback_cc = 20
        self._filter_freq_cc = 21
        self._filter_res_cc = 22
        
        # Transport controls (buttons)
        self._play_note = 36
        self._stop_note = 37
        self._record_note = 38
        
        logger.info("Initialized Yaeltex controller mappings")
    
    def process_message(self, message: MIDIMessage) -> None:
        """Process Yaeltex MIDI message."""
        if message.message_type == MIDIMessageType.CONTROL_CHANGE:
            self._handle_yaeltex_cc(message)
        elif message.message_type in [MIDIMessageType.NOTE_ON, MIDIMessageType.NOTE_OFF]:
            self._handle_yaeltex_note(message)
    
    def _handle_yaeltex_cc(self, message: MIDIMessage) -> None:
        """Handle Yaeltex control change."""
        cc_number = message.controller_number
        cc_value = message.normalized_controller_value
        
        if cc_number in self._volume_ccs:
            # Volume fader
            channel = self._volume_ccs.index(cc_number)
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": f"channel_{channel}_volume", "value": cc_value},
                source="MIDI:Yaeltex"
            )
        elif cc_number == self._pan_cc:
            # Pan control (convert 0-1 to -1 to 1)
            pan_value = (cc_value * 2.0) - 1.0
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": "pan", "value": pan_value},
                source="MIDI:Yaeltex"
            )
        elif cc_number == self._reverb_room_cc:
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": "reverb", "room_size": cc_value},
                source="MIDI:Yaeltex"
            )
        elif cc_number == self._reverb_damping_cc:
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": "reverb", "damping": cc_value},
                source="MIDI:Yaeltex"
            )
        elif cc_number == self._delay_time_cc:
            # Map to delay time (0-2 seconds)
            delay_time = cc_value * 2.0
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": "delay", "delay_time": delay_time},
                source="MIDI:Yaeltex"
            )
        elif cc_number == self._delay_feedback_cc:
            # Map to delay feedback (0-0.9)
            feedback = cc_value * 0.9
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": "delay", "feedback": feedback},
                source="MIDI:Yaeltex"
            )
        elif cc_number == self._filter_freq_cc:
            # Map to filter frequency (20-20000 Hz, log scale)
            freq = 20.0 * (1000.0 ** cc_value)  # Log scale
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": "filter", "frequency": freq},
                source="MIDI:Yaeltex"
            )
        elif cc_number == self._filter_res_cc:
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": "filter", "resonance": cc_value},
                source="MIDI:Yaeltex"
            )
        
        logger.debug(f"Yaeltex CC {cc_number} = {cc_value}")
    
    def _handle_yaeltex_note(self, message: MIDIMessage) -> None:
        """Handle Yaeltex note message."""
        note_number = message.note_number
        
        if message.message_type == MIDIMessageType.NOTE_ON and message.velocity > 0:
            if note_number == self._play_note:
                self.event_manager.emit(
                    EventType.SOUND_PLAY_START,
                    {"triggered_by": "yaeltex_play_button"},
                    source="MIDI:Yaeltex"
                )
            elif note_number == self._stop_note:
                self.event_manager.emit(
                    EventType.SOUND_PLAY_STOP,
                    {"triggered_by": "yaeltex_stop_button"},
                    source="MIDI:Yaeltex"
                )
            elif note_number == self._record_note:
                self.event_manager.emit(
                    EventType.MIDI_MESSAGE,
                    {"action": "toggle_record", "triggered_by": "yaeltex_record_button"},
                    source="MIDI:Yaeltex"
                )
        
        logger.debug(f"Yaeltex note {note_number} {message.message_type.value}")


class AkaiMidimixController(MIDIController):
    """Akai Midimix controller with predefined mappings."""
    
    def __init__(self, event_manager: EventManager):
        super().__init__("AkaiMidimix", event_manager)
        self._setup_default_mappings()
    
    def _setup_default_mappings(self) -> None:
        """Set up default Akai Midimix mappings."""
        # Track faders (16-23)
        self._track_faders = list(range(16, 24))
        
        # Track knobs upper row (0-7)
        self._upper_knobs = list(range(0, 8))
        
        # Track knobs lower row (8-15)
        self._lower_knobs = list(range(8, 16))
        
        # Master fader
        self._master_fader = 62
        
        # Transport buttons
        self._rec_arm_buttons = list(range(1, 9))  # 1-8
        self._solo_buttons = list(range(17, 25))   # 17-24
        self._mute_buttons = list(range(33, 41))   # 33-40
        
        logger.info("Initialized Akai Midimix controller mappings")
    
    def process_message(self, message: MIDIMessage) -> None:
        """Process Akai Midimix MIDI message."""
        if message.message_type == MIDIMessageType.CONTROL_CHANGE:
            self._handle_midimix_cc(message)
        elif message.message_type in [MIDIMessageType.NOTE_ON, MIDIMessageType.NOTE_OFF]:
            self._handle_midimix_note(message)
    
    def _handle_midimix_cc(self, message: MIDIMessage) -> None:
        """Handle Midimix control change."""
        cc_number = message.controller_number
        cc_value = message.normalized_controller_value
        
        if cc_number in self._track_faders:
            # Track volume
            track = self._track_faders.index(cc_number)
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": f"track_{track}_volume", "value": cc_value},
                source="MIDI:AkaiMidimix"
            )
        elif cc_number in self._upper_knobs:
            # Upper knobs - map to various effects
            knob = self._upper_knobs.index(cc_number)
            if knob == 0:  # Pan
                pan_value = (cc_value * 2.0) - 1.0
                self.event_manager.emit(
                    EventType.AUDIO_STATE_CHANGE,
                    {"parameter": "pan", "value": pan_value},
                    source="MIDI:AkaiMidimix"
                )
            elif knob == 1:  # Reverb room size
                self.event_manager.emit(
                    EventType.AUDIO_STATE_CHANGE,
                    {"parameter": "reverb", "room_size": cc_value},
                    source="MIDI:AkaiMidimix"
                )
            elif knob == 2:  # Reverb damping
                self.event_manager.emit(
                    EventType.AUDIO_STATE_CHANGE,
                    {"parameter": "reverb", "damping": cc_value},
                    source="MIDI:AkaiMidimix"
                )
        elif cc_number in self._lower_knobs:
            # Lower knobs - more effects
            knob = self._lower_knobs.index(cc_number)
            if knob == 0:  # Delay time
                delay_time = cc_value * 2.0
                self.event_manager.emit(
                    EventType.AUDIO_STATE_CHANGE,
                    {"parameter": "delay", "delay_time": delay_time},
                    source="MIDI:AkaiMidimix"
                )
            elif knob == 1:  # Delay feedback
                feedback = cc_value * 0.9
                self.event_manager.emit(
                    EventType.AUDIO_STATE_CHANGE,
                    {"parameter": "delay", "feedback": feedback},
                    source="MIDI:AkaiMidimix"
                )
        elif cc_number == self._master_fader:
            # Master volume
            self.event_manager.emit(
                EventType.AUDIO_STATE_CHANGE,
                {"parameter": "volume", "value": cc_value},
                source="MIDI:AkaiMidimix"
            )
        
        logger.debug(f"Midimix CC {cc_number} = {cc_value}")
    
    def _handle_midimix_note(self, message: MIDIMessage) -> None:
        """Handle Midimix note message."""
        note_number = message.note_number
        
        if message.message_type == MIDIMessageType.NOTE_ON and message.velocity > 0:
            if note_number in self._rec_arm_buttons:
                track = self._rec_arm_buttons.index(note_number)
                self.event_manager.emit(
                    EventType.MIDI_MESSAGE,
                    {"action": f"toggle_track_{track}_record"},
                    source="MIDI:AkaiMidimix"
                )
            elif note_number in self._solo_buttons:
                track = self._solo_buttons.index(note_number)
                self.event_manager.emit(
                    EventType.MIDI_MESSAGE,
                    {"action": f"toggle_track_{track}_solo"},
                    source="MIDI:AkaiMidimix"
                )
            elif note_number in self._mute_buttons:
                track = self._mute_buttons.index(note_number)
                self.event_manager.emit(
                    EventType.MIDI_MESSAGE,
                    {"action": f"toggle_track_{track}_mute"},
                    source="MIDI:AkaiMidimix"
                )
        
        logger.debug(f"Midimix note {note_number} {message.message_type.value}")


def create_controller(controller_type: str, event_manager: EventManager) -> MIDIController:
    """Create MIDI controller instance.
    
    Args:
        controller_type: Type of controller ("generic", "yaeltex", "midimix")
        event_manager: Event manager instance
        
    Returns:
        MIDI controller instance
    """
    if controller_type.lower() == "yaeltex":
        return YaeltexController(event_manager)
    elif controller_type.lower() in ["midimix", "akai", "akai_midimix"]:
        return AkaiMidimixController(event_manager)
    else:
        return GenericMIDIController(controller_type, event_manager)