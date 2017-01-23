#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <deque>
#include <algorithm>
#include <numeric>

#include "base/logging.h"
#include "base/stringprintf.h"
#include "emulator.h"
#include "galaxian.h"

using std::deque;

int64 Now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * (int64)1000000 + tv.tv_usec;
}

class Timer {
 public:
  explicit Timer(string name) : name_(name), start_(Now()) {}

  ~Timer() {
    int64 end = Now();
    LOG(INFO) << name_ << ": " << (end - start_) << " " << start_ << " " << end;
  }

 private:
  const string name_;
  const time_t start_;
};

double Random() {
  return (double)random() / RAND_MAX;
}

double Average(const deque<int>& x) {
  return (double)std::accumulate(x.begin(), x.end(), 0) / x.size();
}

int Median(const deque<int>& x) {
  auto x1 = x;
  std::sort(x1.begin(), x1.end());
  return x1[x1.size()/2];
}

namespace galaxian {

class Server {
 public:
  explicit Server(int port) : port_(port) {}

  void Loop() {
    LOG(INFO) << "Running Galaxian server";

    InitSocket();
    RecvStart();

    SkipMenu();

    vector<uint8> beginning;
    Emulator::Save(&beginning);

    vector<uint8> reload;
    Emulator::Save(&reload);

    CHECK(GetState().lifes == 2);

    int prev_score = -1;
    int reward_sum = 0;
    int max_score = 0;
    int max_level = 0;
    deque<int> episode_rewards;

    for (int step = 1; ; ++step) {
      if (Random() < 0.01) {
        Emulator::Save(&reload);
      }

      if (GetLevel() > max_level && reward_sum > 2000) {
        max_level = GetLevel();
        Emulator::Save(&beginning);
        LOG(INFO) << "Level " << max_level;
      }

      uint8 input;
      int seq;
      RecvInput(&input, &seq);
      CHECK_EQ(step, seq);

      bool terminal = false;
      int reward = 0;

      for (int i = 0; i < 5; ++i) {
        Emulator::Step(input);
        if (IsDead()) {
          reward = 0;
          terminal = true;
          break;
        }
      }

      const State s = GetState();
      if (!terminal && prev_score >= 0) {
        reward = s.score - prev_score;
      }
      Respond(seq, s, reward, terminal, input);

      if (!terminal) {
        prev_score = s.score;
        reward_sum += reward;
        max_score = std::max(max_score, reward_sum);
      } else {
        Emulator::Load(Random() < 0.2 ? &beginning : &reload);
        episode_rewards.push_back(reward_sum);
        if (episode_rewards.size() > 100) {
          episode_rewards.pop_front();
        }
        LOG(INFO) << " Seq " << seq << " Max level: " << max_level
                  << " Max rewards: " << max_score << " Score: " << s.score
                  << " rewards: " << reward_sum << " last 100 avg: "
                  << Average(episode_rewards) << " last 100 median: "
                  << Median(episode_rewards);
        prev_score = -1;
        reward_sum = 0;
      }
    }
  }

 private:
  void InitSocket() {
    sockaddr_in sin;
    sin.sin_addr.s_addr = htonl(INADDR_ANY);
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port_);
    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    CHECK_NE(-1, sockfd_) << "INVALID socket";
    CHECK(::bind(sockfd_, (sockaddr*)&sin, sizeof(sin)) == 0)
        << "Failed to bind to port " << port_;
    listen(sockfd_, 1);

    socklen_t sizeof_sin = sizeof(sin);
    sockfd_ = accept(sockfd_, (sockaddr*)&sin, &sizeof_sin);
    CHECK_NE(-1, sockfd_) << "Sock accept failed";
  }

  void RecvBuffer() {
    buffer_.resize(256);
    ssize_t size = recv(sockfd_, &buffer_[0], buffer_.size(), 0);
    CHECK_GT(size, 0);
    buffer_.resize(size);
    buffer_.pop_back();
  }

  void RecvStart() {
    do {
      RecvBuffer();
    } while (buffer_ != "galaxian:start");

    buffer_ = "ack";
    SendBuffer();
  }

  void RecvInput(uint8* input, int* seq) {
    //Timer timer("Recv");
    RecvBuffer();
    char action;
    sscanf(buffer_.c_str(), "%c %d", &action, seq);
    *input = ToInput(action);
  }

  void Respond(int seq, const State& s, int reward, bool terminal,
               uint8 input) {
    //Timer timer("Respond");
    buffer_.clear();
    AppendInt(seq);
    AppendInt(reward);
    AppendInt(terminal ? 1 : 0);
    AppendChar(ToAction(input));
    AppendPoint(s.galaxian);
    AppendPoint(s.missile);
    for (int e : s.still_enemies_encoded) {
      AppendInt(e);
    }
    AppendInt(s.incoming_enemies.size());
    for (const pair<int, Point>& e : s.incoming_enemies) {
      AppendInt(e.first);
      AppendPoint(e.second);
    }
    AppendInt(s.bullets.size());
    for (const pair<int, Point>& b : s.bullets) {
      AppendInt(b.first);
      AppendPoint(b.second);
    }
    SendBuffer();
  }

  void AppendInt(int i) {
    StringAppendF(&buffer_, "%d ", i);
  }

  void AppendChar(char c) {
    StringAppendF(&buffer_, "%c ", c);
  }

  void AppendPoint(const Point& p) {
    StringAppendF(&buffer_, "%d %d ", p.x, p.y);
  }

  void SendBuffer() {
    buffer_.push_back('\n');
    CHECK_GE(send(sockfd_, buffer_.data(), buffer_.size(), 0), 0)
        << "Send failed";
    buffer_.clear();
  }

  const int port_;
  int sockfd_;
  string buffer_;
};

}  // namespace galaxian

int main(int argc, char *argv[]) {
  CHECK_EQ(3, argc);

  CHECK(Emulator::Initialize(argv[1]));

  const int port = atoi(argv[2]);
  galaxian::Server server(port);
  server.Loop();

  Emulator::Shutdown();
  FCEUI_Kill();

  return 0;
}
