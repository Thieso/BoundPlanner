{
  description = "BoundPlanner";

  nixConfig = {
    substituters = [
      "https://nix-community.cachix.org"
      "https://cache.nixos.org/"
      "https://ros.cachix.org"
    ];
    trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo="
    ];
  };

  inputs = {
    ros-flake.url = "github:lopsided98/nix-ros-overlay";
    nixpkgs.follows = "ros-flake/nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      ros-flake,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ ros-flake.overlays.default ];
        config.allowUnfree = true;
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        shellHook = ''
          source "$(get_ros2_path).bash"
          export QT_QPA_PLATFORM=xcb
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
            with pkgs;
            lib.makeLibraryPath [
              xorg.libX11
              xorg.libXt
              xorg.libSM
              zlib
              glib
              udev
              libGL
              glfw
              boost
              libusb1
              gmp
              cddlib
            ]
          }:${pkgs.stdenv.cc.cc.lib}/lib"
            source .venv/bin/activate
            export PYTHONPATH="$PYTHONPATH:$VIRTUAL_ENV/lib/python3.12/site-packages"
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBNVIDIA_PATH"
            unset QT_PLUGIN_PATH
        '';
        name = "BoundPlanner";
        buildInputs =
          with pkgs;
          with pkgs.rosPackages.humble;
          [
            (buildEnv {
              paths = [
                ipopt
                eigen
                pyright
                ruff
                ruff-lsp
                cddlib
                boost
                ros-core
                ros2run
                ros2launch
                colcon
                geometry-msgs
                sensor-msgs
                rviz2
                xacro
                robot-state-publisher
                joint-state-publisher
                joint-state-publisher-gui
                ament-cmake-core
                python-cmake-module
                rosidl-default-generators
                ament-cmake
                std-msgs
                geometry-msgs

                python312Packages.pip
                python312Packages.tkinter
                python312Packages.cmake
                python312Packages.isort
                python312Packages.debugpy
                python312Packages.pinocchio
                python312Packages.matplotlib
                python312Packages.coal
                python312Packages.eigenpy
                python312Packages.isort
                python312Packages.debugpy
                cmake
                gcc
                stdenv
                udev
                libGL
                glfw
                graphviz
                gmp
              ];
            })
          ];
      };
    };
}
