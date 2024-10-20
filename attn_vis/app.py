from flask import Flask, render_template, jsonify, request
import torch
import argparse
import sys

app = Flask(__name__)

# Initialize generated_info as None; it will be loaded based on the command-line argument
generated_info = None

def load_generated_info(info_file):
    global generated_info
    try:
        generated_info = torch.load(info_file, map_location='cpu')
        print(f"Successfully loaded '{info_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{info_file}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading '{info_file}': {e}")
        sys.exit(1)

def preprocess_generated_info():
    """
    Preprocesses the generated_info to handle empty tokens and calculate total attention received.
    Also assigns display tokens and flags for generated tokens.
    """
    global generated_info
    initial_context_length = None
    for info in generated_info:
        if 'initial_context_length' in info:
            initial_context_length = info['initial_context_length']
            break

    # Preprocess generated_info to handle empty tokens and calculate total attention received
    total_tokens = len(generated_info)
    for idx, info in enumerate(generated_info):
        token_text = info.get('decoded_token', '')
        if not token_text.strip():
            info['display_token'] = '___'
        else:
            info['display_token'] = token_text
        info['index'] = idx  # Store index for reference
        info['is_generated'] = not info.get('is_initial_context', False)

        # Total attention received per layer and head is precomputed in generate function
        # For overall opacity, we can sum over layers and heads
        total_attention = sum(
            sum(layer_attention) for layer_attention in info.get('total_attention_per_layer_head', [])
        )
        info['total_attention_received'] = total_attention

@app.route('/')
def index():
    tokens = [{'index': idx, 'token_id': info['token_id'], 'decoded_token': info['display_token'], 'is_generated': info['is_generated']}
              for idx, info in enumerate(generated_info)]
    if not tokens:
        return "No tokens available to display.", 400
    num_layers = len(generated_info[0]['attn_info_per_layer']) if 'attn_info_per_layer' in generated_info[0] else 0
    num_heads = len(generated_info[0]['attn_info_per_layer'][0]['q_norms']) if num_layers > 0 else 0
    return render_template('index.html', tokens=tokens, num_layers=num_layers, num_heads=num_heads)

@app.route('/get_token_info', methods=['POST'])
def get_token_info():
    token_index = int(request.form.get('token_index', -1))
    selected_layer = int(request.form.get('selected_layer', 0))
    selected_head = int(request.form.get('selected_head', 0))

    if token_index < 0 or token_index >= len(generated_info):
        return jsonify({'error': 'Invalid token index.'}), 400

    token_info = generated_info[token_index]
    attn_info_layer = token_info['attn_info_per_layer'][selected_layer]
    token_norms = token_info['token_norms']

    is_initial_context = token_info.get('is_initial_context', False)

    # Extract next_token_probs with decoded tokens for the selected layer
    next_token_probs_per_layer = token_info.get('next_token_probs_per_layer', [])
    if selected_layer < len(next_token_probs_per_layer):
        next_token_probs = next_token_probs_per_layer[selected_layer]
    else:
        next_token_probs = []

    total_tokens = len(generated_info)

    # Extract data for the selected head
    topk_indices_to = attn_info_layer.get('topk_indices_to', [])[selected_head]
    topk_values_to = attn_info_layer.get('topk_values_to', [])[selected_head]

    # Prepare attention_from_selected array for highlighting (Attended To)
    attention_from_selected = [0.0] * total_tokens
    for idx_to, val in zip(topk_indices_to, topk_values_to):
        idx_to = int(idx_to)
        if idx_to < total_tokens:
            attention_from_selected[idx_to] = val.item()

    # Prepare attention_to_selected array for highlighting (Attending From)
    attention_to_selected = [0.0] * total_tokens

    # Define get_context_tokens function before calling it
    def get_context_tokens(idx, selected_idx):
        context_tokens = []
        context_size = 5  # Number of tokens on each side
        start = idx - context_size
        end = idx + context_size + 1
        for idx_ctxt in range(start, end):
            if 0 <= idx_ctxt < total_tokens:
                token_text = generated_info[idx_ctxt].get('display_token', ' ')
                is_center = (idx_ctxt == idx)
                is_selected = (idx_ctxt == selected_idx)
                context_tokens.append({'token': token_text, 'is_center': is_center, 'is_selected': is_selected})
            else:
                # Handle out-of-range indices
                context_tokens.append({'token': ' ', 'is_center': False, 'is_selected': False})
        return context_tokens

    # Get context tokens for the selected token
    selected_token_context = get_context_tokens(token_index, token_index)

    # Get top tokens attending to this token for the selected head
    topk_tokens_from = []
    topk_scores_from = []
    topk_distances_from = []
    topk_q_norms_from = []

    attention_scores = token_info.get('attention_scores', {})
    top_tokens_attending_to_list = attention_scores.get('top_tokens_attending_to', [])
    if len(top_tokens_attending_to_list) > selected_layer and len(top_tokens_attending_to_list[selected_layer]) > selected_head:
        top_tokens_attending_to = top_tokens_attending_to_list[selected_layer][selected_head]
        for idx_from, score in top_tokens_attending_to:
            if idx_from < total_tokens:
                attention_to_selected[idx_from] = score
                context_tokens = get_context_tokens(idx_from, token_index)
                topk_tokens_from.append({
                    'token_index': idx_from,
                    'decoded_token': generated_info[idx_from].get('decoded_token', ''),
                    'context_tokens': context_tokens
                })
                topk_scores_from.append(score)
                distance = idx_from - token_index
                topk_distances_from.append(distance)
                # Get q_norm of the token attending to this token
                q_norm = generated_info[idx_from]['token_norms']['q_norms'][selected_layer][selected_head].item()
                topk_q_norms_from.append(q_norm)

    # Get top tokens attended to by this token
    topk_tokens_to = []
    topk_scores_to = []
    topk_distances_to = []
    topk_k_norms_to = []
    topk_v_norms_to = []

    for idx_to, val in zip(topk_indices_to, topk_values_to):
        idx_to = int(idx_to)
        if idx_to < total_tokens and val.item() > 0:
            context_tokens = get_context_tokens(idx_to, token_index)
            topk_tokens_to.append({
                'token_index': idx_to,
                'decoded_token': generated_info[idx_to].get('decoded_token', ''),
                'context_tokens': context_tokens
            })
            topk_scores_to.append(val.item())
            distance = idx_to - token_index
            topk_distances_to.append(distance)
            # Get k_norm and v_norm of the token attended to
            k_norm = generated_info[idx_to]['token_norms']['k_norms'][selected_layer][selected_head].item()
            v_norm = generated_info[idx_to]['token_norms']['v_norms'][selected_layer][selected_head].item()
            topk_k_norms_to.append(k_norm)
            topk_v_norms_to.append(v_norm)

    # Get norms for selected token
    selected_embedding_norm = attn_info_layer['embedding_norms'][selected_head].item()
    selected_q_norm = attn_info_layer['q_norms'][selected_head].item()
    selected_k_norm = attn_info_layer['k_norms'][selected_head].item()
    selected_v_norm = attn_info_layer['v_norms'][selected_head].item()
    weighted_v_norm = attn_info_layer['weighted_v_norms'][selected_head].item()
    weighted_v_excl_topk_norm = attn_info_layer['weighted_v_excl_topk_norms'][selected_head].item()

    # Total attention falling on selected token
    total_attention = token_info['total_attention_per_layer_head'][selected_layer][selected_head]

    # Prepare total attention per token for opacity adjustment
    total_attention_per_token = [0.0] * total_tokens
    for idx, info in enumerate(generated_info):
        total_attention_per_token[idx] = info['total_attention_per_layer_head'][selected_layer][selected_head]

    response = {
        'is_initial_context': is_initial_context,
        'selected_token_text': token_info['display_token'],
        'selected_token_context': selected_token_context,
        'topk_tokens_to': topk_tokens_to,
        'topk_scores_to': topk_scores_to,
        'topk_distances_to': topk_distances_to,
        'topk_k_norms_to': topk_k_norms_to,
        'topk_v_norms_to': topk_v_norms_to,
        'topk_tokens_from': topk_tokens_from,
        'topk_scores_from': topk_scores_from,
        'topk_distances_from': topk_distances_from,
        'topk_q_norms_from': topk_q_norms_from,
        'attention_from_selected': attention_from_selected,
        'attention_to_selected': attention_to_selected,
        'selected_embedding_norm': selected_embedding_norm,
        'selected_q_norm': selected_q_norm,
        'selected_k_norm': selected_k_norm,
        'selected_v_norm': selected_v_norm,
        'weighted_v_norm': weighted_v_norm,
        'weighted_v_excl_topk_norm': weighted_v_excl_topk_norm,
        'total_attention_on_selected': total_attention,
        'total_attention_per_token': total_attention_per_token,
        'next_token_probs': next_token_probs  # Include next token probabilities with decoded tokens
    }
    return jsonify(response)

def main():
    parser = argparse.ArgumentParser(description="Flask App for Attention Visualization")
    parser.add_argument('info_file', type=str, help='Path to the generated info file (e.g., test.info)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the Flask app on')
    parser.add_argument('--port', type=int, default=5005, help='Port to run the Flask app on')
    args = parser.parse_args()

    load_generated_info(args.info_file)
    preprocess_generated_info()

    # Run the Flask app with the specified host and port
    app.run(debug=True, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
