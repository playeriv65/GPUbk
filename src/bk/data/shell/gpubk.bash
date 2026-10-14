# SPDX-License-Identifier: Apache-2.0
# Source this file explicitly; GPUBK never edits shell startup files.
[[ -n ${BASH_VERSION-} ]] || return 0
[[ ${GPUBK_REFRESH_IN_PROGRESS-} != 1 ]] || return 0

_gpubk_hook_path=${BASH_SOURCE[0]}

# A new noninteractive Bash still runs its previous BASH_ENV startup file.
if [[ $- != *i* && -z ${_gpubk_previous_env_loaded-} ]]; then
    _gpubk_previous_env_loaded=1
    if [[ -n ${GPUBK_PREVIOUS_BASH_ENV-} && ${GPUBK_PREVIOUS_BASH_ENV} != "$_gpubk_hook_path" ]]; then
        source "$GPUBK_PREVIOUS_BASH_ENV"
    fi
fi

bk_gpu_refresh() {
    local previous_status=$? visible
    export CUDA_VISIBLE_DEVICES=''
    if visible=$(GPUBK_REFRESH_IN_PROGRESS=1 command bk env --value 2>/dev/null); then
        if [[ -z $visible || $visible =~ ^(GPU-|MIG-)[A-Za-z0-9_./:-]+(,(GPU-|MIG-)[A-Za-z0-9_./:-]+)*$ ]]; then
            export CUDA_VISIBLE_DEVICES=$visible
        fi
    fi
    return "$previous_status"
}

if [[ ${BASH_ENV-} != "$_gpubk_hook_path" ]]; then
    export GPUBK_PREVIOUS_BASH_ENV=${BASH_ENV-}
fi
export BASH_ENV=$_gpubk_hook_path

if [[ $- == *i* && -z ${_gpubk_prompt_installed-} ]]; then
    if [[ $(declare -p PROMPT_COMMAND 2>/dev/null) == 'declare -a '* ]]; then
        PROMPT_COMMAND=(bk_gpu_refresh "${PROMPT_COMMAND[@]}")
    elif [[ -n ${PROMPT_COMMAND-} ]]; then
        PROMPT_COMMAND=(bk_gpu_refresh "$PROMPT_COMMAND")
    else
        PROMPT_COMMAND=(bk_gpu_refresh)
    fi
    _gpubk_prompt_installed=1
fi

bk_gpu_refresh
unset _gpubk_hook_path
