# Copyright (c) 2015-2016, Maciej Sieczka, Gregory M. Kurtzer. All rights
# reserved.
#
# Copyright (c) 2017-2018, SyLabs, Inc. All rights reserved.
# Copyright (c) 2017, SingularityWare, LLC. All rights reserved.
#
# Copyright (c) 2015-2017, Gregory M. Kurtzer. All rights reserved.
#
# Minimal installation process is defined in
# libexec/bootstrap/modules-v2/dist-arch.sh. A couple extra actions are called
# from here in `%post' section. Adjust them as needed.
# https://wiki.archlinux.org/index.php/Installation_Guide may come in handy.

Bootstrap: docker
From: archlinux/base

%runscript
    echo "This is what happens when you run the container..."

%files

%environment
    #export LC_ALL=C

%labels
    AUTHOR adrien.gruson@gmail.com


%post
    echo "Hello from inside the container"

    # Set time zone. Use whatever you prefer instead of UTC.
    # ln -s /usr/share/zoneinfo/UTC /etc/localtime

    # Set locale. Use whatever you prefer instead of en_US.
    echo 'en_US.UTF-8 UTF-8' > /etc/locale.gen
    locale-gen
    echo 'LANG=en_US.UTF-8' > /etc/locale.conf
    # Mind that Singularity's shell will use host's locale no matter what
    # anyway, as of version 2.1.2. This may change in a future release.

    # Set the package mirror server(s). This is only for the output image's
    # mirrorlist. `pacstrap' can only use your hosts's package mirrors.
    echo 'Server = http://arch.mirror.constant.com/$repo/os/$arch' > /etc/pacman.d/mirrorlist
    # Add any number of fail-over servers, eg:
    echo 'Server = http://mirror.sfo12.us.leaseweb.net/archlinux/$repo/os/$arch' >> /etc/pacman.d/mirrorlist

    # This is the list of common package between GCC and clang build
    pacman -Sy --noconfirm bash-completion cmake fftw libpng jasper zlib xerces-c xorg glew git awk python3 make python-pip eigen 
    # We need to install base-devel for GCC or clang (contain gcc)
    pacman -Sy --noconfirm base-devel rust embree boost openexr

    # Remove the packages downloaded to image's Pacman cache dir.
    pacman -Sy --noconfirm pacman-contrib
    paccache -r -k0
