# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=
# User specific aliases and functions
echo "IMPORT CONDA"
alias conda="sudo /root/anaconda3/bin/conda"
alias python="sudo /root/anaconda3/bin/python"
echo "FINISH CONDA COMMAND"
alias gunicorn="sudo /root/anaconda3/bin/gunicorn"
export PATH="$PATH:/root/anaconda3/bin/python"
