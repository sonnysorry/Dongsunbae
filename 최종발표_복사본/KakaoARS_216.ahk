#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
#SingleInstance Force
#Persistent
#Warn
SendMode Input  ; Recommended for new scripts due to its superior speed and reliability.
SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.
SetTitleMatchMode, 3
CoordMode, ToolTip, Screen
DetectHiddenText, On
DetectHiddenWindows, On

;-- 변수 선언 ----------------------------------------------------------------------------------------------------
global varVer = "KakaoARS"
global varOnoff := 0
global varStatus1, varStatus2, varStatus3
global varWorking
global varAway
global varHome
global varNTF
global var5min
global varEdit
global varTerm = "none"

;=================================================================================================================================
;=================================================================================================================================
start:

IfNotExist, MAssistant.ini
{
	FileRead, var, text1.txt
	FileRead, var2, text2.txt
	IniWrite, %var% , MAssistant.ini, Sample,varWorking
	IniWrite, %var2%, MAssistant.ini, Sample,varAway
	
}
va := ""
IniRead, varStatus2		, MAssistant.ini, Status,varStatus2,0
;-- 현재 상태 ----------------------------------------------------------------------------------------------------
Gui, Add, GroupBox, x12 y10 w90 h140 , 현재 상태
Gui, Add, Radio, x22 y50 w60 h20 gactionStatus vvarStatus1, 존댓말
Gui, Add, Radio, x22 y90 w70 h20 gactionStatus vvarStatus2, 반말


IniRead, varStatus1		, MAssistant.ini, Status,varStatus1,1
IniRead, varStatus2		, MAssistant.ini, Status,varStatus2,0

GuiControl,,varStatus1, %varStatus1%
GuiControl,,varStatus2, %varStatus2%


;-- 에디트 컨트롤 -------------------------------------------------------------------------------------------------
Gui, Add, GroupBox, x112 y10 w210 h140 , 현재 지정된 메시지
Gui, Add, Edit, x122 y30 w190 h110 vvarEdit

IniRead, varWorking		, MAssistant.ini, Sample,varWorking
IniRead, varAway		, MAssistant.ini, Sample,varAway



Gui, Submit, NoHide
If varStatus1 = 1
	GuiControl,,varEdit, %varWorking%
else if varStatus2 = 1
	GuiControl,,varEdit, %varAway%


;--버튼 모음-------------------------------------------------------------------------------------------------------
Gui Font, cBlack
GuiControl Font, status
Gui, Add, text, x335 y40 w60 h50 center hidden vstatus, 자동 응답 중
Gui, Add, Button, x335 y90 w60 h30, 도움말
Gui, Add, Button, x335 y125 w60 h30, 종료
Gui, Show, x30 y30 w410 h160, KakaoARS


main:

Gui +lastfound
hWnd := WinExist()

DllCall( "RegisterShellHookWindow", UInt,hWnd )
MsgNum := DllCall( "RegisterWindowMessage", Str,"SHELLHOOK" )

inactivity_limit=300	; measured in seconds
how_often_to_test=10	; measured in seconds
show_tooltip=1       ; 1=show, anything else means hide

inactivity_limit_ms:=inactivity_limit*1000
how_often_to_test_ms:=how_often_to_test*1000

IfNotExist, %A_ScriptDir%\customers.txt
			FileAppend, KakaoARS Customer Log`n,%A_ScriptDir%\customers.txt

settimer, check_active, %how_often_to_test_ms%

; ===========================================================================

; 초기 데이터를 위한 대화내용 저장

dataupdate := 0 

WinShow, 카카오톡
WinActivate, 카카오톡

ControlClick, x32 y117, 카카오톡 ; 메시지 목록 클릭
sleep, 500 ; 매크로가 겹치는 오류를 방지하기 위한 sleep 시간 (1000 = 1초)
yl := 150 ; (메시지 목록의 y좌표 값)

loop, 5
{
	WinActivate, ahk_class EVA_Window_Dblclk
	Sleep, 500
	ControlClick, x300 y%yl%, ahk_class EVA_Window_Dblclk
	Sleep, 500
	Send, {Enter}
	Sleep, 500
	Send, ^{s}
    Sleep, 1000
	ControlClick, Edit2, 다른 이름으로 저장
    Sleep, 500
	Send, %A_ScriptDir%
	Sleep, 500
	Send, {Enter}
	Sleep, 500
	Send, {Enter}
	Sleep, 3000
	Send, {Esc}
	Sleep, 1000
	Send, {Esc}
	Sleep, 500
	WinShow, 카카오톡
    WinActivate, 카카오톡
	
	yl := yl + 75
	
}

;==============================================
;텍스트파일 merge
Send, #r
sleep, 1000
Send, cmd
sleep, 1000
Send, {Enter}
sleep, 2000

Send, cd %A_ScriptDir%
Send, {Enter}
sleep, 1000

Send, type *.txt > merge.txt
Send, {Enter}
sleep, 1000
Send, exit
Send, {Enter}
Sleep, 500

dataupdate :=1

; 데이터 분석 파이썬 파일 실행
IfExist %A_ScriptDir%\merge.txt
{
	; Run, %A_ScriptDir%\main.py
	Send, #r
	Send, cmd
	Send, {Enter}
	Send, %A_ScriptDir%\main.py
	Send, {Enter}
	IfExist %A_ScriptDir%\db.txt
		dataupdate := 1
}



;================================================

if %dataupdate%
{
	Sleep, 500
	Critical
	Gui, Submit, NoHide
	OnMessage(MsgNum, (varOnOff := !varOnOff) ? "ShellMessage" : "")

    If GetKeyState("Ctrl", "P") = 0
	{
		if CheckKakaoLogin()
			Intro()
	}
	
	GuiControl, show, status
    GuiControl, disable, varStatus1
    GuiControl, disable, varStatus2
    GuiControl, disable, varStatus3
    GuiControl, disable, varNTF
    GuiControl, disable, var5min
    GuiControl, disable, msgSave
    GuiControl, disable, about
    GuiControl, disable, varEdit
    TrayTip,KakaoARS, 자동 응답을 시작합니다,1
    SetTimer, RemoveTrayTip, 1000
	
}


check_active:
if A_TimeIdlePhysical > %inactivity_limit_ms%
{
	If varOnoff = 0
	{
		If var5min = 1
		{
			ControlClick, On/Off, KakaoARS
		}
	}
}
return

Intro() ;5초 대기 후 시작
{

	GuiControl, enable, onoff
}

Outro()
{
	TrayTip,KakaoARS, 자동 응답을 종료합니다^^,1
	SetTimer, RemoveTrayTip, 1000
	SaveAll()
}

CheckKakaoLogin() ;카카오톡 로그인 상태 확인
{
	IfWinExist, 카카오톡
	{
		WinShow, 카카오톡
		WinActivate, 카카오톡
		ControlGet, varCKLcontrol, Visible,, Edit2, 카카오톡
		If varCKLcontrol{
			msgbox, 0x1040,KakaoARS,카카오톡에 로그인을 해 주세요
			varOnoff := 0
			return 0
		}
	}
	else{
		varOnoff := 0
		msgbox, 0x1040, KakaoARS,카카오톡을 먼저 실행 해 주세요 
	return 0
	}
	ControlClick, x330 y15, ahk_class EVA_Window_Dblclk
	return 1
}


ShellMessage( wParam,lParam ) ;푸시 알림 감지 및 채팅창 열기
{
	WinGetTitle, Title, ahk_id %lParam%
	If (Title = "카카오톡") 
	{
		BlockInput, On
		Winwait, ahk_class EVA_Window_Dblclk,,0
		ControlClick, x1 y1, ahk_class EVA_Window_Dblclk
		sleep, 100
		If DetermineFriend()
			return
		
		IfNotExist, %A_ScriptDir%\customers.txt
			FileAppend, KakaoARS Customer Log`n,%A_ScriptDir%\customers.txt

		WinGet, id, list, ahk_class #32770,,카카오톡
		Loop %id%
		{
			if %id%
			{
				
				this_id := id%A_Index%
				WinGetTitle, varCTitle, ahk_id %this_id%
				i = 1
				j = 2
				
				Loop
			    {
					FileReadLine, line, %A_ScriptDir%\db.txt, %i%
                    FileReadLine, lin, %A_ScriptDir%\db.txt, %j%
                    if (ErrorLevel = 1)
					{
						break
					}
		
                    if (varCTitle = line)
					{
						varTerm = %lin%
						break
					}
					
					i += 2
					j += 2                    
                }
				
				varWrite := 1
				;--------중복 여부만을 검사---------------------------------
                Loop, read, %A_ScriptDir%\customers.txt
                {
					If varCTitle = %A_LoopReadLine%
						varWrite := 0
                }
                if %varWrite%
                {
					if (varTerm = "1")
                    {
						FileReadLine, msgtext, %A_ScriptDir%\text1.txt, 1
                        FileAppend, %varCTitle%`n, %A_ScriptDir%\customers.txt
                        SendKaKaoMessage(msgtext,varCTitle)
					}
                    else if (varTerm = "0")
                    {
						FileReadLine, msgtext, %A_ScriptDir%\text2.txt, 1
                        FileAppend, %varCTitle%`n, %A_ScriptDir%\customers.txt
                        SendKaKaoMessage(msgtext,varCTitle)
                    }
					else
					{
						FileAppend, %varCTitle%`n, %A_ScriptDir%\customers.txt
						SendKaKaoMessage(varEdit,varCTitle)
					}
						
						
				}
		    }
			
		}
	}
	BlockInput, Off	
	return
}

DetermineFriend() ;친구여부 판단, 1 : 친구, 0 : 친구아님
{
	Winwait, ahk_class EVA_Window_Dblclk,,0
	ControlClick, x20 y20, ahk_class EVA_Window_Dblclk

	ControlGetPos, varDFnd,,,,EVA_Window1,ahk_class #32770
	If varDFnd
	{
		Send, {ESC}
		return 1
	}
	else
		return 0
}

; 사용예시 SendKakaoMessage("Message","Matthew Burrows")
SendKaKaoMessage(Word, Name) ;카톡으로 메시지 보내기
{
	Clipboard=%Word%
	clipWait
	IfWinExist, %Name%
	{
		PostMessage,0x302,1,0,RichEdit50W1,%Name%
        sleep,120
        PostMessage,0x100,0x0D,0,RichEdit50W1,%Name%
		Sleep, 500
		savebin := 1
		Loop, read, %A_ScriptDir%\db.txt
		{
			If Name = %A_LoopReadLine%
			{
				savebin := 0
				break
			}
		}
		if %savebin%
		{
			Send, ^{s}
            Sleep, 1000
			ControlClick, Edit2, 다른 이름으로 저장
            Sleep, 500
	        Send, %A_ScriptDir%
	        Sleep, 500
	        Send, {Enter}
	        Sleep, 500
	        Send, {Enter}
	        Sleep, 3000
	        
	        
			
		}
		
		Sleep, 500
		WinShow, ahk_class #32770
        WinActivate, ahk_class #32770
		Sleep, 500
		ControlClick, x285 y21, ahk_class #32770
		
	}
}

SaveAll() ;설정내용 저장
{
	Gui, Submit, nohide
	IniWrite, %varStatus1%, MAssistant.ini, Status,varStatus1
	IniWrite, %varStatus2%, MAssistant.ini, Status,varStatus2
	
	If varStatus1 = 1
	IniWrite, %varEdit%, MAssistant.ini, Sample, varWorking
	Else if varStatus2 = 1
	IniWrite, %varEdit%, MAssistant.ini, Sample, varAway
	
	
}

actionStatus:
Gui, Submit, NoHide
IniRead, varWorking		, MAssistant.ini, Sample,varWorking
IniRead, varAway		, MAssistant.ini, Sample,varAway

If varStatus1 = 1
	GuiControl,,varEdit, %varWorking%
else if varStatus2 = 1
	GuiControl,,varEdit, %varAway%
return

actionNTF:
action5min:
Gui, Submit, NoHide
return

Button종료:
GuiClose:
Outro()
FileDelete, customers.txt
ExitApp

Button도움말:
Gui,2:Add, Edit, x10 y10 w310 h180, - 프로그램 진행과정 -`n`n1. 최근 5개의 대화내용을 저장 후 merge.txt로 합친다.`n`n2. 메시지가 왔을 때 보낸 사람의 이름이 db.txt에 있는지 확인한 후, 존댓말 혹은 반말을 해야하는 상대이거나 이름이 db.txt에 없는 사람인지에 따라 메시지를 보낸다.`n`n3. 메시지를 보낼 때 db.txt에 이름이 없던 사람이라면, 그 사람과의 대화내용을 저장한다`n`n* 사용시 주의사항 : `n프로그램을 시작하고, 카카오톡 메시지창을 최소화해야 메시지가 보내진다. 이외에 카카오톡 대화창을 활성화시키면 자동으로 메시지가 보내지지 않는다.
return

RemoveToolTip:
SetTimer, RemoveToolTip, Off
ToolTip
return

RemoveTrayTip:
SetTimer, RemoveTrayTip, Off
TrayTip
return