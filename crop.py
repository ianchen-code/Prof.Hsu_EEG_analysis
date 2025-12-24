import mne
import numpy as np
import os

# ========================================
# CONFIGURATION - CHANGE THESE
# ========================================
base_path = '/Volumes/USB/EEG research/Data3'
file_numbers = range(41, 46)  # 41 to 45 inclusive
file_pattern = "Function-{:03d}.cnt"  # e.g., Function-041.cnt

# Output folder for cropped segments
output_folder = os.path.join(base_path, "cropped_segments")
os.makedirs(output_folder, exist_ok=True)

# ========================================
# PROCESS EACH FILE
# ========================================
all_segments_info = []

for file_num in file_numbers:
    file_name = file_pattern.format(file_num)
    file_path = os.path.join(base_path, file_name)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: {file_name} not found, skipping...")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {file_name}")
    print(f"{'='*60}")
    
    # Read the .cnt file (try ANT Neuro format first, then Neuroscan)
    try:
        raw = mne.io.read_raw_ant(file_path, preload=True)
        print("Loaded as ANT Neuro CNT format")
    except:
        try:
            raw = mne.io.read_raw_cnt(file_path, preload=True)
            print("Loaded as Neuroscan CNT format")
        except Exception as e:
            print(f"Could not read file: {e}")
            continue
    
    # Get events/triggers
    events, event_dict = mne.events_from_annotations(raw)
    
    # Extract trigger codes and sample numbers
    trigger_samples = events[:, 0]
    trigger_codes = events[:, 2]
    
    # Get the reverse mapping (event_id -> description)
    id_to_desc = {v: k for k, v in event_dict.items()}
    
    # Convert trigger codes to integers (they're stored as string descriptions)
    trigger_codes_int = []
    for event_id in trigger_codes:
        desc = id_to_desc[event_id]
        try:
            trigger_codes_int.append(int(desc))
        except ValueError:
            trigger_codes_int.append(-1)  # For non-numeric triggers
    trigger_codes_int = np.array(trigger_codes_int)
    
    print(f"Total events: {len(trigger_codes_int)}")
    print(f"Unique triggers: {np.unique(trigger_codes_int)}")
    print(f"First 10 triggers: {trigger_codes_int[:10]}")
    
    # Find segments between trigger 61-120 and trigger 200/201
    segments = []
    
    i = 0
    while i < len(trigger_codes_int):
        # Check if current trigger is in range 61-120
        if 61 <= trigger_codes_int[i] <= 120:
            start_sample = trigger_samples[i]
            start_trigger = trigger_codes_int[i]
            
            # Look for the next 200 or 201 trigger
            j = i + 1
            while j < len(trigger_codes_int):
                if trigger_codes_int[j] in [200, 201]:
                    end_sample = trigger_samples[j]
                    end_trigger = trigger_codes_int[j]
                    
                    # Store segment info
                    segments.append({
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'start_trigger': start_trigger,
                        'end_trigger': end_trigger,
                        'start_time': start_sample / raw.info['sfreq'],
                        'end_time': end_sample / raw.info['sfreq'],
                        'duration': (end_sample - start_sample) / raw.info['sfreq']
                    })
                    
                    i = j  # Move to the end trigger
                    break
                j += 1
        i += 1
    
    # Print found segments for this file
    print(f"\nFound {len(segments)} segments in {file_name}:")
    for idx, seg in enumerate(segments, 1):
        print(f"  Segment {idx}: trigger {seg['start_trigger']} -> {seg['end_trigger']}, "
              f"duration: {seg['duration']:.3f}s")
    
    # Crop and save each segment
    for idx, seg in enumerate(segments, 1):
        # Crop the raw data
        cropped = raw.copy().crop(
            tmin=seg['start_time'],
            tmax=seg['end_time']
        )
        
        # Create filename with file number and segment info
        output_file = os.path.join(
            output_folder, 
            f"Function-{file_num:03d}_seg{idx}_trig{seg['start_trigger']}-{seg['end_trigger']}.fif"
        )
        
        cropped.save(output_file, overwrite=True)
        print(f"  Saved: Function-{file_num:03d}_seg{idx}_trig{seg['start_trigger']}-{seg['end_trigger']}.fif")
        
        # Store info for summary
        all_segments_info.append({
            'file_name': file_name,
            'file_num': file_num,
            'segment_idx': idx,
            'start_trigger': seg['start_trigger'],
            'end_trigger': seg['end_trigger'],
            'duration': seg['duration'],
            'output_file': os.path.basename(output_file)
        })

# ========================================
# SUMMARY
# ========================================
print(f"\n{'='*60}")
print("PROCESSING COMPLETE - SUMMARY")
print(f"{'='*60}")
print(f"Output folder: {output_folder}")
print(f"Total segments extracted: {len(all_segments_info)}")

# Group by file
from collections import defaultdict
by_file = defaultdict(list)
for info in all_segments_info:
    by_file[info['file_name']].append(info)

print(f"\nBreakdown by file:")
for file_name, infos in sorted(by_file.items()):
    print(f"\n  {file_name}: {len(infos)} segments")
    for info in infos:
        print(f"    - Segment {info['segment_idx']}: "
              f"trigger {info['start_trigger']}->{info['end_trigger']}, "
              f"duration: {info['duration']:.3f}s")

print(f"\nAll cropped files saved to: {output_folder}")