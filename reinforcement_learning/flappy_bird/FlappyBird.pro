#-------------------------------------------------
#
# Project created by QtCreator 2018-11-10T19:56:13
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FlappyBird
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
    ai.cpp \
    ai_thread.cpp \
    bird.cpp \
    game_logic.cpp \
    game_object.cpp \
    game_painter.cpp \
    game_widget.cpp \
    land.cpp \
    main_window_game.cpp \
    main_window_init.cpp \
    main_window_ui.cpp \
    pipe.cpp \
    ui_pushbutton.cpp

HEADERS += \
    ai.h \
    ai_param.h \
    ai_thread.h \
    base.h \
    bird.h \
    game_logic.h \
    game_object.h \
    game_painter.h \
    game_widget.h \
    land.h \
    main_window.h \
    matrix.h \
    pipe.h \
    ui_count_param.h \
    ui_pushbutton.h \
    ui_size_param.h \
    ui_style_sheet.h

QMAKE_CXXFLAGS +=  -Wno-unused-parameter

CONFIG += C++11
