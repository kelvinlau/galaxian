#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <netinet/in.h>

#include "cc-lib/base/logging.h"
#include "cc-lib/base/stringprintf.h"
#include "simplefm2.h"
#include "emulator.h"
#include "galaxian.h"

int64 Now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * (int64)1000000 + tv.tv_usec;
}

class Timer {
 public:
  explicit Timer(string name) : name_(name), start_(Now()) {}

  ~Timer() {
    cerr << name_ << ": " << (Now() - start_) << "\n";
  }

 private:
  const string name_;
  const time_t start_;
};

namespace galaxian {

class Server {
 public:
  void Loop() {
    cout << "Running Galaxian server\n";
    cout.flush();

    InitSocket();

    SkipMenu();

    vector<uint8> beginning;
    Emulator::Save(&beginning);

    CHECK(GetState().lifes == 2);

    int prev_score = 0;
    int max_score = 0;

    for (int step = 0; ; ++step) {
      //cout << "\n";
      //Timer timer(StringPrintf("Step %d", step));

      uint8 input;
      int seq;
      RecvInput(&input, &seq);

      for (int i = 0; i < 12; ++i) {
        Emulator::Step(input);
      }

      const State s = GetState();
      const int reward = s.score - prev_score;
      Respond(seq, s, reward);

      prev_score = s.score;
      max_score = std::max(max_score, s.score);

      if (s.lifes < 2) {
        Emulator::Load(&beginning);
        prev_score = 1000;  // Next respond will have reward = -1000.
      }

      if (step % 1000 == 0) {
        cout << "Step " << step << " Max score: " << max_score << "\n";
      }
    }
  }

 private:
  void InitSocket() {
    sockaddr_in sin;
    sin.sin_addr.s_addr = htonl(INADDR_ANY);
    sin.sin_family = AF_INET;
    sin.sin_port = htons(62343);
    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd_ == -1) {
      cerr << "INVALID socket\n";
      abort();
    }
    if (bind(sockfd_, (sockaddr *)&sin, sizeof(sin))) {
      cerr << "Failed to bind\n";
      abort();
    }
    listen(sockfd_, 1);

    socklen_t sizeof_sin = sizeof(sin);
    sockfd_ = accept(sockfd_, (sockaddr*)&sin, &sizeof_sin);
    if (sockfd_ == -1) {
      cerr << "Sock accept failed\n";
      abort();
    }
  }

  void RecvInput(uint8* input, int* seq) {
    //Timer timer("Recv");
    buffer_.resize(256);
    ssize_t size = recv(sockfd_, &buffer_[0], buffer_.size(), 0);
    if (size <= 0) {
      cerr << "Recv size: " << size << "\n";
      abort();
    }
    buffer_.resize(size);
    // cerr << "Recv: " << buffer_ << "\n";
    char action;
    sscanf(buffer_.c_str(), "%c %d", &action, seq);
    *input = ToInput(action);
  }

  void Respond(int seq, const State& s, int reward) {
    //Timer timer("Respond");
    buffer_.clear();
    AppendInt(seq);
    AppendInt(reward);
    AppendPoint(s.galaxian);
    AppendPoint(s.missile);
    for (int e : s.still_enemies_encoded) {
      AppendInt(e);
    }
    AppendInt(s.incoming_enemies.size());
    for (const Point& e : s.incoming_enemies) {
      AppendPoint(e);
    }
    AppendInt(s.bullets.size());
    for (const Point& b : s.bullets) {
      AppendPoint(b);
    }
    SendBuffer();
  }

  void AppendInt(int i) {
    StringAppendF(&buffer_, "%d ", i);
  }

  void AppendPoint(const Point& p) {
    StringAppendF(&buffer_, "%d %d ", p.x, p.y);
  }

  void SendBuffer() {
    buffer_.push_back('\n');
    //cerr << "Send: " << buffer_.size() << " bytes\n";
    if (send(sockfd_, buffer_.data(), buffer_.size(), 0) < 0) {
      fprintf(stderr, "Send failed\n");
      abort();
    }
    buffer_.clear();
  }

  int sockfd_;
  string buffer_;
};

}  // namespace galaxian

int main(int argc, char *argv[]) {
  //cout.sync_with_stdio(false);

  Emulator::Initialize("galaxian.nes");

  galaxian::Server server;
  server.Loop();

  Emulator::Shutdown();

  // exit the infrastructure
  FCEUI_Kill();

  fprintf(stderr, "SUCCESS.\n");
  return 0;
}
