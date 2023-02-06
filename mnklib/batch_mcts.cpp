#include <cstdint>
#include <vector>
#include <iostream>
#include <array>
#include <random>

#include "mcts_inv.h"

using State885 = State<8, 8, 5>;

struct Game
{
    Game() : buf(5000 * 8 * 8), temp(4.0), rollouts(5000), gen{rd()}, mcts(buf), rollouts_left(rollouts)
    {
    }

    void end_rollout(double value_to_record)
    {
        mcts.record(node_id, value_to_record);
        rollouts_left--;
    }

    void simulate_until_expand(void (*log_freq_cb)(), void (*log_game_done_cb)(), int32_t *log_boards_buffer, float *log_probs_buffer)
    {   
        while (true)
        {
            if (rollouts_left == 0)
            {
                make_move(log_freq_cb, log_game_done_cb, log_boards_buffer, log_probs_buffer);
                continue;
            }

            search_state = state; // copy here
            node_id = mcts.root_id;
            while (mcts.nodes[node_id].children_count > 0)
            {
                node_id = mcts.select(mcts.nodes[node_id], temp);
                search_state.apply_move(mcts.nodes[node_id].in_action);
            }

            last_player = 1 - search_state.player;

            if (!search_state.finished())
            {
                return;
            }

            // being here means we got to end of game and can get exact score
            double value = search_state.score(last_player);
            end_rollout(value);
        }
    }

    void restart()
    {
        state = State885();
    }

    uint64_t get_move_index()
    {
        static thread_local std::array<uint64_t, 64> visits;
        std::fill(visits.begin(), visits.end(), 0ull);
        for (int i = 0; i < mcts.nodes[mcts.root_id].children_count; i++)
        {
            const auto &node = mcts.nodes[mcts.nodes[mcts.root_id].children_from + i];
            visits[node.in_action] = node.N;
        }

        // TODO: configure 10
        if (state.stones_played() > 8) {
            // pick greedily
            auto it = std::max_element(visits.begin(), visits.end());
            return std::distance(visits.begin(), it);
        } else {
            // sample 
            std::discrete_distribution<> d(visits.begin(), visits.end());
            return d(gen);
        }
    }

    void fill_visits(float *freq_buffer) {
        for (int i = 0; i < 64; i++) {
            freq_buffer[i] = 0.0f;
        }
        for (int i = 0; i < mcts.nodes[mcts.root_id].children_count; i++)
        {
            const auto &node = mcts.nodes[mcts.nodes[mcts.root_id].children_from + i];
            freq_buffer[node.in_action] = node.N;
        }
    }

    void make_move(void (*log_freq_cb)(), void (*log_game_done_cb)(), int32_t *log_boards_buffer, float *log_probs_buffer)
    {
        // logging training data here
        state.fill_boards(log_boards_buffer);
        fill_visits(log_probs_buffer);
        log_freq_cb();

        // applying move
        uint64_t index = get_move_index();
        state.apply_move(index);
        state.p();
        
        rollouts_left = rollouts;
        mcts.reset();

        if (state.finished())
        {
            if (log_game_done_cb != nullptr)
            {
                log_game_done_cb();
            }
            restart();
        }
    }

    // global scope
    std::vector<MCTSNode> buf;
    double temp;
    int rollouts;
    std::random_device rd;
    std::mt19937 gen;

    // per game instance scope
    State885 state;

    // per each move scope
    MCTS mcts;

    // within current search state scope
    int rollouts_left;
    State885 search_state;
    int node_id;
    int last_player;
    double value_to_record;
};

void expand_and_eval(std::vector<Game> &games, void (*eval_cb)(), int32_t *boards_buffer, float *probs_buffer)
{
    static const int kBoardElements = 2 * 8 * 8;
    static const int kProbElements = 8 * 8;
    // std::cout << "expand again!" << std::endl;
    for (size_t i = 0; i < games.size(); i++)
    {
        auto &g = games[i];
        g.mcts.nodes[g.node_id].children_from = g.mcts.size;
        if (eval_cb != nullptr)
        {
            g.search_state.fill_boards(boards_buffer + i * kBoardElements);
        }
    }
    // eval_cb takes boards_buffer as an input and writes results to probs_buffer
    // probs_buffer is filled with 1.0 originally
    if (eval_cb != nullptr)
    {
        eval_cb();
    }

    for (size_t i = 0; i < games.size(); i++)
    {
        auto &g = games[i];
        uint64_t moves = g.search_state.valid_actions();
        int j = g.mcts.size;
        for (uint64_t k = 0; k < 8 * 8; k++)
        {
            if ((1ull << k) & moves)
            {
                g.mcts.nodes[j] = MCTSNode(probs_buffer[i * kProbElements + k]);
                g.mcts.nodes[j].parent = g.node_id;
                g.mcts.nodes[j].in_action = k;
                j++;
            }
        }
        g.mcts.nodes[g.node_id].children_count = j - g.mcts.size;
        g.mcts.size = j;

        while (!g.search_state.finished())
        {
            g.search_state.take_random_action();
        }

        double value = g.search_state.score(g.last_player);
        g.end_rollout(value);
    }
}

extern "C"
{
    void batch_mcts(uint32_t batch_size, int32_t *boards_buffer, float *probs_buffer, int32_t *log_boards_buffer, float *log_probs_buffer, void (*eval_cb)(), void (*log_freq_cb)(), void (*log_game_done_cb)())
    {
        std::vector<Game> games{batch_size};
        while (true)
        {
            for (auto &g : games)
            {
                g.simulate_until_expand(log_freq_cb, log_game_done_cb, log_boards_buffer, log_probs_buffer);
            }
            expand_and_eval(games, eval_cb, boards_buffer, probs_buffer);
        }
    }
}