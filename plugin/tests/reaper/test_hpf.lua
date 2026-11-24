-- Prueba del plugin en REAPER, sin sonido. En una pestaña de proyecto nueva: una pista con un
-- fragmento real y el plugin recién insertado; un render offline (el plugin escucha solo durante el
-- render), la lectura de lo que aplicó y un segundo render con eso aplicado. Primero el AU, después
-- el VST3, y al final el VST3 recién insertado con un render a 44.1 kHz (si el dispositivo corre a
-- otra frecuencia, REAPER vuelve a preparar el plugin al empezar y al terminar el render).
-- carpeta de trabajo: la del script, que debe tener gtr_line.wav (audio que no va en el repositorio)
local dir = ({reaper.get_action_context()})[2]:match("^(.*[/\\])")
local out = io.open(dir .. "resultado.txt", "w")
local function log(...)
  local t = {}
  for _, v in ipairs({...}) do t[#t + 1] = tostring(v) end
  out:write(table.concat(t, " "), "\n")
  out:flush()
end

local RENDER = 42230   -- File: Render project, using the most recent render settings, auto-close render dialog
local ok, name = pcall(reaper.kbd_getTextFromCmd, RENDER, 0)
log("REAPER", reaper.GetAppVersion(), "| accion", RENDER, ok and name or "?")
local _, devrate = reaper.GetAudioDeviceInfo("SRATE", "")
log("frecuencia del dispositivo", devrate)

reaper.Main_OnCommand(40859, 0)   -- File: New project tab (no toca otros proyectos abiertos)
reaper.InsertTrackAtIndex(0, true)
local tr = reaper.GetTrack(0, 0)
reaper.SetOnlyTrackSelected(tr)
reaper.SetEditCurPos(0, false, false)
reaper.InsertMedia(dir .. "gtr_line.wav", 0)
log("items en la pista", reaper.CountTrackMediaItems(tr))

local function exists(path)
  local f = io.open(path, "rb")
  if f then f:close() return true end
  return false
end

local function params(fx, tag)
  local found = {}
  for p = 0, reaper.TrackFX_GetNumParams(tr, fx) - 1 do
    local _, pn = reaper.TrackFX_GetParamName(tr, fx, p, "")
    local _, pv = reaper.TrackFX_GetFormattedParamValue(tr, fx, p, "")
    if pn == "Cutoff" or pn == "Slope" or pn == "Filter" then
      log(" ", tag, pn, pv)
      found[pn] = p
    end
  end
  return found
end

local function render(pattern, srate)
  reaper.GetSetProjectInfo_String(0, "RENDER_FILE", dir, true)
  reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", pattern, true)
  reaper.GetSetProjectInfo_String(0, "RENDER_FORMAT", "evaw", true)   -- WAV con su configuración por defecto
  reaper.GetSetProjectInfo(0, "RENDER_SETTINGS", 0, true)             -- mezcla del master
  reaper.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 1, true)           -- todo el proyecto
  reaper.GetSetProjectInfo(0, "RENDER_SRATE", srate, true)
  reaper.GetSetProjectInfo(0, "RENDER_CHANNELS", 1, true)
  reaper.GetSetProjectInfo(0, "RENDER_NORMALIZE", 0, true)
  reaper.GetSetProjectInfo(0, "RENDER_DITHER", 0, true)
  local t = reaper.time_precise()
  reaper.Main_OnCommand(RENDER, 0)
  log(" ", pattern, exists(dir .. pattern .. ".wav") and "renderizado" or "NO SE ESCRIBIO", string.format("(%.1f s)", reaper.time_precise() - t))
end

-- una pasada: inserta el plugin, renderiza, espera a que aplique (hasta 30 s), lee y vuelve a
-- renderizar si se pide; al terminar llama a nextStep con el índice del plugin
local function pass(fxname, prefix, srate, second, nextStep)
  local fx = reaper.TrackFX_AddByName(tr, fxname, false, -1)
  local _, shown = reaper.TrackFX_GetFXName(tr, fx, "")
  log(prefix .. ":", fxname, "->", fx, shown)
  if fx < 0 then nextStep(-1) return end
  local idx = params(fx, "antes")
  local _, before = reaper.TrackFX_GetFormattedParamValue(tr, fx, idx["Cutoff"] or 0, "")
  render(prefix .. "_1", srate)
  local t0 = reaper.time_precise()
  local function waitApply()
    local _, now = reaper.TrackFX_GetFormattedParamValue(tr, fx, idx["Cutoff"] or 0, "")
    local waited = reaper.time_precise() - t0
    if now == before and waited < 30 then reaper.defer(waitApply) return end
    log(" ", now == before and "no aplico nada en 30 s" or string.format("aplico a los %.1f s", waited))
    params(fx, "despues")
    if second then render(prefix .. "_2", srate) end
    nextStep(fx)
  end
  reaper.defer(waitApply)
end

local function finish(fx)
  reaper.Main_SaveProjectEx(0, dir .. "prueba.rpp", 0)
  if fx >= 0 then reaper.TrackFX_Show(tr, fx, 3) end   -- ventana flotante, para verla
  log("fin")
  out:close()
end

pass("AU: HPF! (Atonal)", "au", 48000, true, function(fx)
  if fx >= 0 then
    reaper.Main_SaveProjectEx(0, dir .. "prueba_au.rpp", 0)
    reaper.TrackFX_Delete(tr, fx)
  end
  pass("VST3: HPF! (Atonal)", "vst3", 48000, true, function(fx2)
    if fx2 >= 0 then
      reaper.Main_SaveProjectEx(0, dir .. "prueba_vst3.rpp", 0)
      reaper.TrackFX_Delete(tr, fx2)
    end
    pass("VST3: HPF! (Atonal)", "vst3_44k", 44100, false, finish)
  end)
end)
