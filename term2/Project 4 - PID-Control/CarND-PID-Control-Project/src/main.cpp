#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main()
{
  uWS::Hub h;

  PID pid;
  PID pid_throttle;
  double cruising_speed_default = 35;
  // TODO: Initialize the pid variable.
/*
  // different test values since my twiddle implementation was not working properly
  pid.Init(0.2, 0.01, 5.0); // ok
  pid.Init(0.01, 0.01, 1.0); // crash
  pid.Init(0.1, 0.005, 10.0); // not ideal as ride is not smooth; steering angle continuously changes to left to right but stays mostly in the middle
  pid.Init(0.9, 0.005, 4.0); // ok
  pid.Init(0.9, 0.01, 6.0); // higher the d, the more the steering angle oscillates
  pid.Init(0.9, 0.01, 2.0); // ok
  pid.Init(0.1, 0.01, 1.0); // crash
  pid.Init(0.1, 0.1, 2.0); // crash
  pid.Init(0.1, 0.01, 0.1); // crash */
  
  /** 
  Observations/Notes:
  * The higher the cruising speed, the more eratic the ride is going to be on the road. At
    lower speeds, the init values are more forgiving. This just reinforces that twiddle algo
	is very beneficial if not necessary in the real-world.
  * Intentionally set cruising speed to 35, but can go even as high as 45 though it does
    become a very jerky ride. A value of 55 is disastrous. Lower than 35 is more pleasant.
  */
  
  pid.Init(0.1, 0.01, 2.0); // best so far, least jerky
  pid_throttle.Init(0.3, 0.0, 0.0);
  h.onMessage([&pid, &pid_throttle, cruising_speed_default](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data).substr(0, length));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          double angle = std::stod(j[1]["steering_angle"].get<std::string>());
          double steer_value;
		  double throttle = 0.0;
          /*
          * TODO: Calcuate steering value here, remember the steering value is
          * [-1, 1].
          * NOTE: Feel free to play around with the throttle and speed. Maybe use
          * another PID controller to control the speed!
          */
          pid.UpdateError(cte);
          /** instead of an if () {...}, just calculate using min/max to keep in range of -1 to 1. */
		  steer_value = std::max(-1.,(std::min(1.,pid.TotalError())));

		  /** 
		  Observations/Notes:
		  * A high throttle value will allow the car to go faster but without using twiddle (or
		    something similar), the car steering will oscillate left to right with large angles
			and quickly go off track.
		  * Revisit twiddle after the course completes, but to complete the project for
            now, I will submit what I have which is not optimized as I would have liked.
          * Used the speed to calculate throttle error since throttle is a constant.
          * Used a speed pid and used speed value and a various constant values to create an error
            value to pass into UpdateError(), but regardless I did not see any advantage or 
			disadvantage.	
		  */
          pid_throttle.UpdateError(speed - cruising_speed_default);
          /** instead of an if () {...}, just calculate using min/max to keep in range of -1 to 1. */
		  throttle = std::max(-1.,(std::min(1.,pid_throttle.TotalError())));
		  
          // DEBUG
          std::cout << "CTE: " << cte << " Steering Value: " << steer_value << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle;
		  msgJson["speed"] = speed;
		  msgJson["angle"] = angle;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
