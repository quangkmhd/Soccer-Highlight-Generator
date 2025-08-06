import json
import math
from datetime import timedelta

def seconds_to_time_str(seconds):
    return str(timedelta(seconds=int(seconds)))

class HighlightExtractor:
    def __init__(self, actions_data, config):

        self.config = config
        raw_actions = actions_data['predictions']

        processed_actions = []
        for action in raw_actions:
            try:
                h, m, s = action['timestamp'].split(':')
                seconds = int(h) * 3600 + int(m) * 60 + float(s)
                processed_actions.append({
                    'timestamp': seconds,
                    'label': action['event'],
                    'confidence': float(action['confidence'])
                })
            except (ValueError, KeyError, TypeError):
                continue

        self.actions = sorted(processed_actions, key=lambda x: x['timestamp'])
        self.action_blocks = []

        self.S_events = set(self.config['events']['s_events'])
        self.G_events = set(self.config['events']['g_events'])
        self.K_events = set(self.config['events']['k_events'])
        self.E_events = set(self.config['events']['e_events'])
        self.F_events = set(self.config['events']['f_events'])
        self.C_events = set(self.config['events']['c_events'])

        self.confidence_thresholds = self.config['filtering'].get('confidence_thresholds', {})

    def _group_consecutive_actions(self):
        time_threshold = 3.0
        
        actions_by_label = {}
        for action in self.actions:
            label = action['label']
            if label not in actions_by_label:
                actions_by_label[label] = []
            actions_by_label[label].append(action)
        
        for label, actions in actions_by_label.items():
            if not actions:
                continue
                
            actions.sort(key=lambda x: x['timestamp'])
            
            current_block = [actions[0]]
            
            for action in actions[1:]:
                if action['timestamp'] - current_block[-1]['timestamp'] <= time_threshold:
                    current_block.append(action)
                else:
                    self.action_blocks.append({
                        'label': label,
                        'start_time': current_block[0]['timestamp'],
                        'end_time': current_block[-1]['timestamp'],
                        'actions': current_block,
                        'count': len(current_block),
                        'avg_confidence': sum(a['confidence'] for a in current_block) / len(current_block)
                    })
                    current_block = [action]
            
            if current_block:
                self.action_blocks.append({
                    'label': label,
                    'start_time': current_block[0]['timestamp'],
                    'end_time': current_block[-1]['timestamp'],
                    'actions': current_block,
                    'count': len(current_block),
                    'avg_confidence': sum(a['confidence'] for a in current_block) / len(current_block)
                })

    def _calculate_block_score(self, block):
        return math.log(block['count'] + 1) * block['avg_confidence']

    def _find_foul_card_highlights(self):
        seq_config = self.config['windows']['foul_card_sequence']

        foul_blocks = [
            b for b in self.action_blocks 
            if b['label'] in self.F_events and b['avg_confidence'] >= self.confidence_thresholds.get(b['label'], 0)
        ]
        card_blocks = [
            b for b in self.action_blocks 
            if b['label'] in self.C_events and b['avg_confidence'] >= self.confidence_thresholds.get(b['label'], 0)
        ]
        valid_pairs = []
        for card_block in card_blocks:
            potential_fouls = [
                foul for foul in foul_blocks
                if 0 < (card_block['start_time'] - foul['start_time']) <= seq_config['max_foul_to_card_duration']
            ]
            
            if not potential_fouls:
                continue

            best_foul_block = max(potential_fouls, key=self._calculate_block_score)
            
            foul_score = self._calculate_block_score(best_foul_block)
            card_score = self._calculate_block_score(card_block)
            pair_score = (foul_score + card_score) / 2

            valid_pairs.append({
                'foul': best_foul_block,
                'card': card_block,
                'score': pair_score
            })
        valid_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        potential_highlights = []
        used_foul_ids = set()
        used_card_ids = set()

        for pair in valid_pairs:
            foul_id = pair['foul']['start_time']
            card_id = pair['card']['start_time']

            if foul_id in used_foul_ids or card_id in used_card_ids:
                continue

            used_foul_ids.add(foul_id)
            used_card_ids.add(card_id)

            foul_block = pair['foul']
            card_block = pair['card']
            
            foul_time = foul_block['start_time']
            s_search_start = seq_config['s_search_start']
            s_search_end = seq_config['s_search_end']
            
            pre_event_blocks = [
                block for block in self.action_blocks
                if foul_time - s_search_end <= block['start_time'] <= foul_time - s_search_start
                and block['label'] in self.S_events
            ]
            if pre_event_blocks:
                best_s_block = max(pre_event_blocks, key=self._calculate_block_score)
                start_time = best_s_block['start_time']
            else:
                start_time = foul_time - seq_config['fallback_pre_window']

            card_time = card_block['start_time']
            post_event_blocks = [
                block for block in self.action_blocks
                if seq_config['post_window_start'] < block['start_time'] - card_time <= seq_config['post_window_end']
                and block['label'] in self.E_events
            ]
            if post_event_blocks:
                first_e_block = min(post_event_blocks, key=lambda block: block['start_time'])
                end_time = first_e_block['start_time']
            else:
                end_time = card_time + seq_config['fallback_post_window']

            events_in_clip = [
                action for action in self.actions       
                if start_time <= action['timestamp'] <= end_time
            ]
            
            primary_event = {
                'label': 'Foul->Card',
                'timestamp': foul_block['start_time'],
                'confidence': pair['score']
            }
            events_in_clip.insert(0, primary_event)

            potential_highlights.append({
                'start_time': start_time,
                'end_time': end_time,
                'events': events_in_clip
            })

        return potential_highlights

    def find_highlights(self):
        potential_highlights = []

        anchor_blocks_candidates = [block for block in self.action_blocks if block['label'] in self.G_events or block['label'] in self.K_events]
        
        # Lọc các anchor block không đủ ngưỡng tin cậy
        anchor_blocks = []
        for block in anchor_blocks_candidates:
            threshold = self.confidence_thresholds.get(block['label'])
            # Nếu không có ngưỡng (is None) thì luôn giữ lại (ví dụ: 'Goal')
            # Nếu có ngưỡng, chỉ giữ lại nếu confidence >= ngưỡng
            if threshold is None or block['avg_confidence'] >= threshold:
                anchor_blocks.append(block)

        g_config = self.config['windows']['g_event']
        k_config = self.config['windows']['k_event']

        for anchor_block in anchor_blocks:
            anchor_time = anchor_block['start_time']
            start_time = None
            end_time = None

            if anchor_block['label'] in self.G_events:
                s_search_start = g_config['s_search_start']
                s_search_end = g_config['s_search_end']
                
                pre_event_blocks = [
                    block for block in self.action_blocks
                    if anchor_time - s_search_end <= block['start_time'] <= anchor_time - s_search_start
                    and block['label'] in self.S_events
                ]
                if pre_event_blocks:
                    best_s_block = max(pre_event_blocks, key=self._calculate_block_score)
                    start_time = best_s_block['start_time']
                else:
                    # Fallback: lùi về fallback_pre_window giây trước G-event
                    start_time = anchor_time - g_config['fallback_pre_window']

                post_event_blocks = [
                    block for block in self.action_blocks
                    if g_config['post_window_start'] < block['start_time'] - anchor_time <= g_config['post_window_end']
                    and block['label'] in self.E_events
                ]
                if post_event_blocks:
                    first_e_block = min(post_event_blocks, key=lambda block: block['start_time'])
                    end_time = first_e_block['start_time']
                else:
                    end_time = anchor_time + g_config['fallback_post_window']

            elif anchor_block['label'] in self.K_events:
                start_time = anchor_time - k_config['pre_window']
                post_event_blocks = [
                    block for block in self.action_blocks
                    if anchor_time < block['start_time'] <= anchor_time + k_config['post_window']
                    and block['label'] in self.E_events
                ]
                if post_event_blocks:
                    first_e_block = min(post_event_blocks, key=lambda block: block['start_time'])
                    end_time = first_e_block['start_time']
                else:
                    end_time = anchor_time + k_config['fallback_post_window']

            if start_time is not None and end_time is not None:
                events_in_clip = [
                    action for action in self.actions 
                    if start_time <= action['timestamp'] <= end_time
                ]
                potential_highlights.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'events': events_in_clip
                })
        return potential_highlights

    def _get_primary_actions_info(self, events):
        priority_order = self.config['ranking']['priority_order']
        primary_actions = []
        
        for label in priority_order:
            matching_events = [e for e in events if e.get('label') == label]
            if matching_events:
                best_event = max(matching_events, key=lambda x: x.get('confidence', 0))
                primary_actions.append({
                    'label': label,
                    'confidence': best_event.get('confidence', 0),
                    'timestamp': best_event.get('timestamp', 0)
                })
        
        return primary_actions

    def _get_clip_primary_event_label(self, clip):
        events = clip.get('events', [])
        if not events:
            return None
        
        primary_event = self._find_primary_event_by_priority(events)
        if primary_event:
            return primary_event.get('label')
        return None

    def _find_primary_event_by_priority(self, events):
        priority_order = self.config['ranking']['priority_order']
        thresholds = self.confidence_thresholds

        for label in priority_order:
            label_events = [e for e in events if e.get('label') == label]
            if not label_events:
                continue

            best_event = max(label_events, key=lambda x: x.get('confidence', 0))

            threshold = thresholds.get(label)
            if threshold is None or best_event.get('confidence', 0) >= threshold:
                return best_event

        return max(events, key=lambda x: x.get('confidence', 0), default=None)

    def _add_clip_metadata(self, clips):
        enhanced_clips = []
        for clip in clips:
            duration = clip['end_time'] - clip['start_time']
            
            events = clip['events']
            
            for event in events:
                if 'timestamp' in event:
                    event['time_str'] = seconds_to_time_str(event['timestamp'])
            
            if events:
                first_action = min(events, key=lambda x: x['timestamp'])
                last_action = max(events, key=lambda x: x['timestamp'])
                
                clip['first_action'] = {
                    'label': first_action['label'],
                    'timestamp': first_action['timestamp'],
                    'time_str': seconds_to_time_str(first_action['timestamp']),
                    'confidence': first_action.get('confidence', 0)
                }
                
                clip['last_action'] = {
                    'label': last_action['label'],
                    'timestamp': last_action['timestamp'],
                    'time_str': seconds_to_time_str(last_action['timestamp']),
                    'confidence': last_action.get('confidence', 0)
                }
            else:
                clip['first_action'] = None
                clip['last_action'] = None
            
            primary_event = self._find_primary_event_by_priority(events)
            if primary_event:
                clip['title'] = primary_event.get('label', 'Unknown Event')
                timestamp = primary_event.get('timestamp')
                clip['action_time'] = seconds_to_time_str(timestamp) if timestamp is not None else None
                clip['confidence'] = primary_event.get('confidence', 0)
            else:
                clip['title'] = 'Unknown Event'
                clip['action_time'] = None
                clip['confidence'] = 0

            clip['start_time_str'] = seconds_to_time_str(clip['start_time'])
            clip['end_time_str'] = seconds_to_time_str(clip['end_time'])
            clip['duration'] = duration
            
            enhanced_clips.append(clip)
        
        return sorted(enhanced_clips, key=lambda x: x['start_time'])

    def _get_event_priority(self, event_label):
        """Lấy độ ưu tiên của event theo ranking config. Số càng nhỏ = ưu tiên càng cao"""
        if event_label is None:
            return 999  # Ưu tiên thấp nhất
        
        priority_order = self.config['ranking']['priority_order']
        try:
            return priority_order.index(event_label)
        except ValueError:
            return len(priority_order)  # Event không có trong ranking

    def _should_merge_or_trim(self, current_label, clip_label):
        """
        Quyết định có nên merge hay trim clips dựa trên rule mới:
        - Goal + Goal = merge
        - Goal + khác = trim (giữ Goal, cắt khác)
        - Khác + Khác (cùng loại) = merge  
        - Khác + Khác (khác loại) = không merge
        """
        if current_label == clip_label:
            return "merge"  # Cùng loại event -> merge
        
        current_priority = self._get_event_priority(current_label)
        clip_priority = self._get_event_priority(clip_label)
        
        # Nếu có Goal thì Goal được ưu tiên
        if current_label == 'Goal' or clip_label == 'Goal':
            return "trim"  # Giữ Goal, cắt event khác
        
        # Nếu không có Goal và khác loại -> không merge
        return "no_merge"

    def _trim_overlapping_clip(self, higher_priority_clip, lower_priority_clip):
        """Cắt bỏ phần overlap của clip có ưu tiên thấp hơn"""
        if lower_priority_clip['start_time'] < higher_priority_clip['end_time']:
            # Cắt đầu clip ưu tiên thấp hơn
            overlap_end = higher_priority_clip['end_time']
            if lower_priority_clip['end_time'] > overlap_end:
                # Còn phần không overlap -> cắt đầu
                lower_priority_clip['start_time'] = overlap_end
                # Lọc lại events trong khoảng thời gian mới
                lower_priority_clip['events'] = [
                    event for event in lower_priority_clip['events']
                    if event.get('timestamp', 0) >= overlap_end
                ]
                return lower_priority_clip
            else:
                # Toàn bộ clip bị overlap -> loại bỏ
                return None
        return lower_priority_clip

    def _merge_overlapping_clips(self, clips):
        if not clips:
            return []

        clips_sorted = sorted(clips, key=lambda x: x['start_time'])
        merged_clips = []

        current_clip = clips_sorted[0]
        current_events = current_clip['events'][:]
        current_primary_label = self._get_clip_primary_event_label(current_clip)
        group = [current_clip]

        for clip in clips_sorted[1:]:
            clip_primary_label = self._get_clip_primary_event_label(clip)
            
            # Kiểm tra overlap
            if clip['start_time'] <= current_clip['end_time']:
                merge_action = self._should_merge_or_trim(current_primary_label, clip_primary_label)
                
                if merge_action == "merge":
                    # Merge hai clips cùng loại
                    current_clip['end_time'] = max(current_clip['end_time'], clip['end_time'])
                    current_events.extend(clip['events'])
                    group.append(clip)
                elif merge_action == "trim":
                    # Trim clip có ưu tiên thấp hơn
                    current_priority = self._get_event_priority(current_primary_label)
                    clip_priority = self._get_event_priority(clip_primary_label)
                    
                    if current_priority < clip_priority:
                        # Current clip ưu tiên cao hơn -> trim clip mới
                        trimmed_clip = self._trim_overlapping_clip(current_clip, clip)
                        if trimmed_clip:
                            # Nếu còn phần không overlap, xử lý như clip riêng biệt
                            clips_sorted.insert(clips_sorted.index(clip) + 1, trimmed_clip)
                    else:
                        # Clip mới ưu tiên cao hơn -> finalize current, bắt đầu mới
                        if len(group) > 1:
                            current_clip['events'] = current_events
                            current_clip['merged_info'] = {
                                'merged_clips_count': len(group),
                                'primary_actions_merged': self._get_primary_actions_info(current_events)
                            }
                        
                        # Trim current clip
                        trimmed_current = self._trim_overlapping_clip(clip, current_clip)
                        if trimmed_current:
                            merged_clips.append(trimmed_current)
                        
                        # Bắt đầu với clip mới
                        current_clip = clip
                        current_events = current_clip['events'][:]
                        current_primary_label = clip_primary_label
                        group = [current_clip]
                else:  # no_merge
                    # Không merge, finalize current clip
                    if len(group) > 1:
                        current_clip['events'] = current_events
                        current_clip['merged_info'] = {
                            'merged_clips_count': len(group),
                            'primary_actions_merged': self._get_primary_actions_info(current_events)
                        }
                    merged_clips.append(current_clip)

                    current_clip = clip
                    current_events = current_clip['events'][:]
                    current_primary_label = clip_primary_label
                    group = [current_clip]
            else:
                # Không overlap, finalize current clip
                if len(group) > 1:
                    current_clip['events'] = current_events
                    current_clip['merged_info'] = {
                        'merged_clips_count': len(group),
                        'primary_actions_merged': self._get_primary_actions_info(current_events)
                    }
                merged_clips.append(current_clip)

                current_clip = clip
                current_events = current_clip['events'][:]
                current_primary_label = clip_primary_label
                group = [current_clip]

        # Xử lý clip cuối cùng
        if len(group) > 1:
            current_clip['events'] = current_events
            current_clip['merged_info'] = {
                'merged_clips_count': len(group),
                'primary_actions_merged': self._get_primary_actions_info(current_events)
            }
        merged_clips.append(current_clip)

        return merged_clips

    def run(self):
        self._group_consecutive_actions()
        
        gk_highlights = self.find_highlights()
        foul_card_highlights = self._find_foul_card_highlights()
        
        potential_highlights = gk_highlights + foul_card_highlights
        
        merged_highlights = self._merge_overlapping_clips(potential_highlights)
        
        final_clips = self._add_clip_metadata(merged_highlights)
        return final_clips

def get_sort_priority(clip, config):
    primary_label = clip.get('title', 'Unknown Event')
    confidence = clip.get('confidence', 0)
    priority_order = config['ranking']['priority_order']
    
    try:
        priority = priority_order.index(primary_label)
    except ValueError:
        priority = len(priority_order)
        
    return (priority, -confidence)

def rank_and_finalize_highlights(highlights, config, limit=30):
    highlights_sorted = sorted(
        highlights,
        key=lambda clip: get_sort_priority(clip, config)
    )
    
    if limit is not None and limit > 0:
        top_highlights = highlights_sorted[:limit]
    else:
        top_highlights = highlights_sorted

    for i, clip in enumerate(top_highlights, 1):
        clip['rank'] = i

    top_highlights_final = sorted(top_highlights, key=lambda x: x['start_time'])
    return top_highlights_final

