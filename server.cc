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

#include "base/logging.h"
#include "base/stringprintf.h"
#include "emulator.h"
#include "galaxian.h"

using std::cerr;
using std::cout;

int64 Now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * (int64)1000000 + tv.tv_usec;
}

string NowStr() {
  using namespace std::chrono;
  system_clock::time_point p = system_clock::now();
  time_t t = system_clock::to_time_t(p);
  string ret = std::ctime(&t);
  ret.pop_back();  // Get rid of '\n'
  return ret;
}

class Timer {
 public:
  explicit Timer(string name) : name_(name), start_(Now()) {}

  ~Timer() {
    int64 end = Now();
    cerr << name_ << ": " << (end - start_) << " " << start_ << " " << end
         << "\n";
  }

 private:
  const string name_;
  const time_t start_;
};

namespace galaxian {

class Server {
 public:
  explicit Server(int port) : port_(port) {}

  void Loop() {
    cout << "Running Galaxian server\n";
    cout.flush();

    InitSocket();

    SkipMenu();

    vector<uint8> beginning;
    Emulator::Save(&beginning);

    vector<uint8> reload;
    Emulator::Save(&reload);

    CHECK(GetState().lifes == 2);

    int prev_score = 0;
    int max_score = 0;

    for (int step = 1; ; ++step) {
      //cout << "\n";
      //Timer timer(StringPrintf("Step %d", step));

      if (random() < 0.01) {
        Emulator::Save(&reload);
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
          reward = -1;
          terminal = true;
          break;
        }
      }

      const State s = GetState();
      if (!terminal) {
        reward = s.score - prev_score;
      }
      Respond(seq, s, reward, terminal, input);
      //cout << "Missile:" << s.missile.x << "," << s.missile.y << "\n";

      prev_score = s.score;
      max_score = std::max(max_score, s.score);

      if (terminal) {
        Emulator::Load(random() < 0.05 ? &beginning : &reload);
        cout << NowStr() << " Step " << step << " Max score: " << max_score
             << " Score: " << s.score << "\n";
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
    if (sockfd_ == -1) {
      cerr << "INVALID socket\n";
      abort();
    }
    if (::bind(sockfd_, (sockaddr *)&sin, sizeof(sin))) {
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
    //cerr << "Send: " << buffer_.size() << " bytes\n";
    if (send(sockfd_, buffer_.data(), buffer_.size(), 0) < 0) {
      fprintf(stderr, "Send failed\n");
      abort();
    }
    buffer_.clear();
  }

  const int port_;
  int sockfd_;
  string buffer_;
};

}  // namespace galaxian

int main(int argc, char *argv[]) {
  //cout.sync_with_stdio(false);

  Emulator::Initialize("galaxian.nes");

  const int port = argc > 1 ? atoi(argv[1]) : 62343;
  galaxian::Server server(port);
  server.Loop();

  Emulator::Shutdown();

  // exit the infrastructure
  FCEUI_Kill();

  fprintf(stderr, "SUCCESS.\n");
  return 0;
}
