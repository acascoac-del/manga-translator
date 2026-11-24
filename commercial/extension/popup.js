const status = document.getElementById('status');
document.getElementById('account').href = API_ORIGIN + '/account/';
async function balance() {
  const response = await chrome.runtime.sendMessage({action:'balance'});
  status.textContent = response.error || `${response.available} créditos disponibles · ${response.reserved} reservados`;
}
chrome.storage.local.get({source:'auto',lastMessage:''}).then(settings => {
  document.getElementById('source').value = settings.source;
  status.textContent = settings.lastMessage;
  balance().catch(() => { status.textContent = 'Ingresá una clave de tu cuenta.'; });
});
document.getElementById('save').addEventListener('click', async () => {
  const token = document.getElementById('token').value.trim();
  if (token.length < 32 || token.length > 100) { status.textContent = 'Clave inválida.'; return; }
  await chrome.storage.local.set({accessToken:token});
  document.getElementById('token').value = '';
  await balance();
});
document.getElementById('logout').addEventListener('click', async () => {
  await chrome.storage.local.remove('accessToken');
  status.textContent = 'Desconectada. Podés revocar la clave desde tu cuenta.';
});
document.getElementById('source').addEventListener('change', event => chrome.storage.local.set({source:event.target.value}));
